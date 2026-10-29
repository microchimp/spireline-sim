"""ctypes bindings to the NativeAOT-compiled Sts2Emulator native library."""

import contextlib
import ctypes
import json
import os
import platform
import signal
import sys
from collections.abc import Callable
from pathlib import Path

_LIB_NAMES = {
    "win32": "Sts2Emulator.dll",
    "linux": "Sts2Emulator.so",
    "darwin": "Sts2Emulator.dylib",
}
_ALLOW_STALE_ENV = "STS2_ALLOW_STALE_NATIVE"
_REQUIRED_NATIVE_API_VERSION = 39
_REQUIRED_RUN_NATIVE_API_VERSION = 32
# The engine this process runs (docs/rebuild/NATIVE_KERNEL.md section 3): read once, when the library loads, and passed
# to Sts2_SetEngine before any handle exists. Unset or "old": the old engine. "core2": the Tier 1 kernel of
# tools/fidelity/migrated_families.json. Inherited by every spawn worker, so one flag on the parent reaches them all.
_ENGINE_ENV = "STS2_ENGINE"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_hint() -> str:
    """Build the rebuild command for the HOST platform.

    These messages hardcoded `win-x64` and Windows-style backslash paths, so
    every stale-library error on macOS or Linux told the reader to build for Windows.
    `scripts/build.sh` already picks the runtime off `uname`, so the advice is one line.
    """
    runtime = {
        "win32": "win-x64",
        "linux": "linux-x64",
        "darwin": "osx-arm64" if platform.machine() == "arm64" else "osx-x64",
    }.get(sys.platform, "win-x64")
    return (
        f"Rebuild the native library with `bash scripts/build.sh {runtime}` "
        f'or `dotnet publish "src/Sts2Emulator/Sts2Emulator.csproj" -c Release '
        f'-r {runtime} --self-contained -o "out"`.'
    )


# The data files the library embeds (the EmbeddedResource items of src/Sts2Emulator/Sts2Emulator.csproj; a test holds the
# two lists equal): an edit to one changes the library's behaviour like an edit to a .cs file does.
_EMBEDDED_DATA_FILES = (
    "tools/fidelity/migrated_families.json",
    "data/id_map.json",
    "tools/fidelity/power_aliases.json",
)


def _native_source_paths(repo_root: Path) -> list[Path]:
    source_root = repo_root / "src" / "Sts2Emulator"
    paths = [
        path
        for path in source_root.rglob("*.cs")
        if "bin" not in path.parts and "obj" not in path.parts
    ]
    paths.extend(source_root.rglob("*.csproj"))
    paths.extend(path for name in _EMBEDDED_DATA_FILES if (path := repo_root / name).is_file())
    return paths


def _assert_native_library_is_fresh(path: Path) -> None:
    if os.environ.get(_ALLOW_STALE_ENV) == "1":
        return

    source_paths = _native_source_paths(_repo_root())
    if not source_paths:
        return

    newest_source = max(source_paths, key=lambda source: source.stat().st_mtime)
    if path.stat().st_mtime >= newest_source.stat().st_mtime:
        return

    raise RuntimeError(
        f"{path} is older than {newest_source}. {_build_hint()} "
        f"Set {_ALLOW_STALE_ENV}=1 only when intentionally testing an older "
        "native build.",
    )


def _assert_native_api_version(lib: ctypes.CDLL, path: Path) -> None:
    try:
        version_func = lib.Sts2_NativeApiVersion
    except AttributeError as exc:
        raise RuntimeError(
            f"{path} does not export Sts2_NativeApiVersion and is too old for "
            f"these Python bindings. {_build_hint()}",
        ) from exc

    version_func.restype = ctypes.c_int
    version_func.argtypes = []
    actual_version = int(version_func())
    if actual_version != _REQUIRED_NATIVE_API_VERSION:
        raise RuntimeError(
            f"{path} exports native API version {actual_version}, but "
            f"sts2_gym requires {_REQUIRED_NATIVE_API_VERSION}. {_build_hint()}",
        )

    try:
        run_version_func = lib.Sts2Run_NativeApiVersion
    except AttributeError as exc:
        raise RuntimeError(
            f"{path} does not export Sts2Run_NativeApiVersion and is too old "
            f"for these Python bindings. {_build_hint()}",
        ) from exc

    run_version_func.restype = ctypes.c_int
    run_version_func.argtypes = []
    actual_run_version = int(run_version_func())
    if actual_run_version != _REQUIRED_RUN_NATIVE_API_VERSION:
        raise RuntimeError(
            f"{path} exports run native API version {actual_run_version}, but "
            f"sts2_gym requires {_REQUIRED_RUN_NATIVE_API_VERSION}. "
            f"{_build_hint()}",
        )


class _DarwinSigaction(ctypes.Structure):
    """macOS's `struct sigaction`, as libc's `sigaction()` takes it."""

    _fields_ = [
        ("handler", ctypes.c_void_p),
        ("mask", ctypes.c_uint32),
        ("flags", ctypes.c_int),
    ]


_SA_SIGINFO = 0x40  # <sys/signal.h> on macOS


def _clear_stale_activation_siginfo() -> None:
    """Clear SA_SIGINFO from a SIGUSR1 that has no handler, before the runtime starts.

    On macOS the NativeAOT runtime stops a thread for the GC by sending it SIGUSR1, and
    its handler then chains to the SIGUSR1 action it found at startup. .NET 9, which
    builds this library, calls that action's `sa_sigaction` whenever its flags hold
    SA_SIGINFO, and rules out SIG_DFL and SIG_IGN only on the other branch
    (dotnet/runtime#132581; fixed by #132900, in .NET 10 from 10.0.13, and not in 9.0).
    macOS resets a caught signal's handler across execve but keeps its flags, so a
    process started by one that had loaded this library -- every runner worker, whose
    parent imports the engine before it spawns them -- inherits SIGUSR1 as
    {SIG_DFL, SA_SIGINFO}. The first background GC that stopped such a worker's main
    thread mid-call jumped to address 0: SIGSEGV, exit -11, intermittently, and only
    once the worker had allocated enough to need one.

    Flags mean nothing on a default or ignored action, so clearing the stale one changes
    nothing but what the runtime reads. A real handler is left alone; the runtime chains
    to that correctly.

    Raises:
        OSError: If SIGUSR1's action cannot be read or rewritten.

    """
    if sys.platform != "darwin":
        return  # Linux clears sa_flags on execve, and its activation signal is SIGRTMIN
    sigaction = ctypes.CDLL(None, use_errno=True).sigaction
    sigaction.restype = ctypes.c_int
    sigaction.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(_DarwinSigaction),
        ctypes.POINTER(_DarwinSigaction),
    ]
    action = _DarwinSigaction()
    if sigaction(signal.SIGUSR1, None, ctypes.byref(action)) != 0:
        raise OSError(ctypes.get_errno(), "sigaction could not read SIGUSR1")
    no_handler = (action.handler or 0) in (signal.SIG_DFL, signal.SIG_IGN)
    if not no_handler or not action.flags & _SA_SIGINFO:
        return
    action.flags &= ~_SA_SIGINFO
    if sigaction(signal.SIGUSR1, ctypes.byref(action), None) != 0:
        raise OSError(ctypes.get_errno(), "sigaction could not rewrite SIGUSR1")


def _load_lib() -> ctypes.CDLL:
    name = _LIB_NAMES.get(sys.platform)
    if name is None:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    search = []
    if "STS2_LIB_PATH" in os.environ:
        search.append(Path(os.environ["STS2_LIB_PATH"]) / name)
    search.append(_repo_root() / "out" / name)
    for path in search:
        if path.exists():
            _assert_native_library_is_fresh(path)
            _clear_stale_activation_siginfo()  # before the runtime reads SIGUSR1
            lib = ctypes.CDLL(str(path))
            _assert_native_api_version(lib, path)
            return lib
    raise FileNotFoundError(
        f"Could not find {name}. Run scripts/build.sh first, or set STS2_LIB_PATH.",
    )


_lib = _load_lib()

# ── function signatures ───────────────────────────────────────────────────────

_lib.Sts2_ObsSize.restype = ctypes.c_int
_lib.Sts2_ObsSize.argtypes = []

_lib.Sts2_MaxEnemies.restype = ctypes.c_int
_lib.Sts2_MaxEnemies.argtypes = []

_lib.Sts2_ObsCardSlotSize.restype = ctypes.c_int
_lib.Sts2_ObsCardSlotSize.argtypes = []

_lib.Sts2_ObsHandOffset.restype = ctypes.c_int
_lib.Sts2_ObsHandOffset.argtypes = []

_lib.Sts2_ObsMaxHand.restype = ctypes.c_int
_lib.Sts2_ObsMaxHand.argtypes = []

_lib.Sts2_ObsPlayerBuffOffset.restype = ctypes.c_int
_lib.Sts2_ObsPlayerBuffOffset.argtypes = []

_lib.Sts2_ObsMaxPlayerBuffs.restype = ctypes.c_int
_lib.Sts2_ObsMaxPlayerBuffs.argtypes = []

_lib.Sts2_ObsMaxEnemyBuffs.restype = ctypes.c_int
_lib.Sts2_ObsMaxEnemyBuffs.argtypes = []

_lib.Sts2_ObsEnemyOffset.restype = ctypes.c_int
_lib.Sts2_ObsEnemyOffset.argtypes = []

_lib.Sts2_ObsEnemySlotSize.restype = ctypes.c_int
_lib.Sts2_ObsEnemySlotSize.argtypes = []

_lib.Sts2_ObsSecondaryIntentOffset.restype = ctypes.c_int
_lib.Sts2_ObsSecondaryIntentOffset.argtypes = []

_lib.Sts2_ObsSelectionKindOffset.restype = ctypes.c_int
_lib.Sts2_ObsSelectionKindOffset.argtypes = []

_lib.Sts2_ObsSelectionCountOffset.restype = ctypes.c_int
_lib.Sts2_ObsSelectionCountOffset.argtypes = []

_lib.Sts2_ObsSelectionOffset.restype = ctypes.c_int
_lib.Sts2_ObsSelectionOffset.argtypes = []

_lib.Sts2_ObsMaxSelectionCandidates.restype = ctypes.c_int
_lib.Sts2_ObsMaxSelectionCandidates.argtypes = []

_lib.Sts2_ObsOrbCapacityOffset.restype = ctypes.c_int
_lib.Sts2_ObsOrbCapacityOffset.argtypes = []

_lib.Sts2_ObsIncomingOffset.restype = ctypes.c_int
_lib.Sts2_ObsIncomingOffset.argtypes = []

_lib.Sts2_ObsMaxIncomingCards.restype = ctypes.c_int
_lib.Sts2_ObsMaxIncomingCards.argtypes = []

_lib.Sts2_ObsOrbOffset.restype = ctypes.c_int
_lib.Sts2_ObsOrbOffset.argtypes = []

_lib.Sts2_ObsMaxOrbs.restype = ctypes.c_int
_lib.Sts2_ObsMaxOrbs.argtypes = []

_lib.Sts2_ObsOrbSlotSize.restype = ctypes.c_int
_lib.Sts2_ObsOrbSlotSize.argtypes = []

_lib.Sts2_Create.restype = ctypes.c_int
_lib.Sts2_Create.argtypes = [ctypes.c_int]

_lib.Sts2_GetRelics.restype = ctypes.c_int
_lib.Sts2_GetRelics.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2_Clone.restype = ctypes.c_int
_lib.Sts2_Clone.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_LoadCombatSnapshot.restype = ctypes.c_int
_lib.Sts2_LoadCombatSnapshot.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_GetStateJson.restype = ctypes.c_int
_lib.Sts2_GetStateJson.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
]

_lib.Sts2_GetSaveJson.restype = ctypes.c_int
_lib.Sts2_GetSaveJson.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
]

_lib.Sts2_GetLastError.restype = ctypes.c_int
_lib.Sts2_GetLastError.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
]

_lib.Sts2_Reset.restype = None
_lib.Sts2_Reset.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

_lib.Sts2_ResetEncounter.restype = None
_lib.Sts2_ResetEncounter.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_ResetEncounterWeak.restype = None
_lib.Sts2_ResetEncounterWeak.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_ResetEncounterAtFloor.restype = None
_lib.Sts2_ResetEncounterAtFloor.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_DebugAddCardToHand.restype = None
_lib.Sts2_DebugAddCardToHand.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_PendingSelectionKind.restype = ctypes.c_int
_lib.Sts2_PendingSelectionKind.argtypes = [ctypes.c_int]

_lib.Sts2_RngCallCount.restype = ctypes.c_int
_lib.Sts2_RngCallCount.argtypes = [ctypes.c_int, ctypes.c_int]

_lib.Sts2_DebugGainMaxHp.restype = None
_lib.Sts2_DebugGainMaxHp.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_ResetEncounterAtFloorWithExtraCards.restype = None
_lib.Sts2_ResetEncounterAtFloorWithExtraCards.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_ResetWithDeck.restype = None
_lib.Sts2_ResetWithDeck.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_ResetWithDeckAndEncounter.restype = None
_lib.Sts2_ResetWithDeckAndEncounter.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2_ResetWithDeckEncounterAndRelics.restype = None
_lib.Sts2_ResetWithDeckEncounterAndRelics.argtypes = [
    ctypes.c_int,  # handle
    ctypes.POINTER(ctypes.c_int),  # deck ids
    ctypes.c_int,  # deck len
    ctypes.c_int,  # encounter id
    ctypes.POINTER(ctypes.c_int),  # relic ids
    ctypes.c_int,  # relic len
    ctypes.POINTER(ctypes.c_int),  # obs buf
]

_lib.Sts2_ResetArena.restype = None
_lib.Sts2_ResetArena.argtypes = [
    ctypes.c_int,  # handle
    ctypes.POINTER(ctypes.c_int),  # deck ids
    ctypes.c_int,  # deck len
    ctypes.POINTER(ctypes.c_int),  # enchantment ids, parallel to the deck
    ctypes.c_int,  # enchantment len
    ctypes.POINTER(ctypes.c_int),  # enchantment amounts, parallel to the deck
    ctypes.c_int,  # enchantment amount len
    ctypes.c_int,  # encounter id (-1 = roll it)
    ctypes.POINTER(ctypes.c_int),  # relic ids
    ctypes.c_int,  # relic len
    ctypes.POINTER(ctypes.c_int),  # potion ids
    ctypes.c_int,  # potion len
    ctypes.c_int,  # player hp
    ctypes.c_int,  # player max hp
    ctypes.c_int,  # player gold
    ctypes.c_int,  # max potion slots (negative: the belt a run at this level starts with)
    ctypes.c_int,  # ascension
    ctypes.c_int,  # total floor
    ctypes.c_int,  # completed combat rooms
    ctypes.POINTER(ctypes.c_int),  # obs
]

_lib.Sts2_CanEnchant.restype = ctypes.c_int
_lib.Sts2_CanEnchant.argtypes = [ctypes.c_int, ctypes.c_int]

_lib.Sts2_DefaultAscension.restype = ctypes.c_int
_lib.Sts2_DefaultAscension.argtypes = []

_lib.Sts2_StartingDeck.restype = ctypes.c_int
_lib.Sts2_StartingDeck.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2_StartingHp.restype = ctypes.c_int
_lib.Sts2_StartingHp.argtypes = [ctypes.c_int]

_lib.Sts2_StartingMaxHp.restype = ctypes.c_int
_lib.Sts2_StartingMaxHp.argtypes = []

_lib.Sts2_StartingPotionSlots.restype = ctypes.c_int
_lib.Sts2_StartingPotionSlots.argtypes = [ctypes.c_int]

_lib.Sts2_Step.restype = ctypes.c_int
_lib.Sts2_Step.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
]

_lib.Sts2_StepTargeted.restype = ctypes.c_int
_lib.Sts2_StepTargeted.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
]

_lib.Sts2_PlayerWon.restype = ctypes.c_int
_lib.Sts2_PlayerWon.argtypes = [ctypes.c_int]

_lib.Sts2_EncounterId.restype = ctypes.c_int
_lib.Sts2_EncounterId.argtypes = [ctypes.c_int]

_lib.Sts2_ActionCount.restype = ctypes.c_int
_lib.Sts2_ActionCount.argtypes = [ctypes.c_int]

_lib.Sts2_ValidActions.restype = None
_lib.Sts2_ValidActions.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2_GetPile.restype = ctypes.c_int
_lib.Sts2_GetPile.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2_GetExecuted.restype = ctypes.c_int
_lib.Sts2_GetExecuted.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2_GetEnemies.restype = ctypes.c_int
_lib.Sts2_GetEnemies.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2_Destroy.restype = None
_lib.Sts2_Destroy.argtypes = [ctypes.c_int]

_lib.Sts2Run_ObsSize.restype = ctypes.c_int
_lib.Sts2Run_ObsSize.argtypes = []

_lib.Sts2Run_MaxActions.restype = ctypes.c_int
_lib.Sts2Run_MaxActions.argtypes = []

_lib.Sts2Run_InfoSize.restype = ctypes.c_int
_lib.Sts2Run_InfoSize.argtypes = []

_lib.Sts2Run_ObsLayout.restype = ctypes.c_int
_lib.Sts2Run_ObsLayout.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]

_lib.Sts2Run_Create.restype = ctypes.c_int
_lib.Sts2Run_Create.argtypes = []
_lib.Sts2Run_ForkCombat.restype = ctypes.c_int
_lib.Sts2Run_ForkCombat.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_lib.Sts2Run_Clone.restype = ctypes.c_int
_lib.Sts2Run_Clone.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_Reset.restype = ctypes.c_int
_lib.Sts2Run_Reset.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int,  # ascension
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_ResetAs.restype = ctypes.c_int
_lib.Sts2Run_ResetAs.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int,  # ascension
    ctypes.c_int,  # character index (RUN_CHARACTERS)
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_PotionActionReason.restype = ctypes.c_int
_lib.Sts2Run_PotionActionReason.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
]

_lib.Sts2Run_SetMaskVersion.restype = ctypes.c_int
_lib.Sts2Run_SetMaskVersion.argtypes = [
    ctypes.c_int,
    ctypes.c_int,  # mask version (RUN_MASK_VERSIONS)
]

_lib.Sts2Run_Step.restype = ctypes.c_int
_lib.Sts2Run_Step.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_StartCombat.restype = ctypes.c_int
_lib.Sts2Run_StartCombat.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_ActionMask.restype = ctypes.c_int
_lib.Sts2Run_ActionMask.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2Run_GetInfo.restype = ctypes.c_int
_lib.Sts2Run_GetInfo.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2Run_GetStateList.restype = ctypes.c_int
_lib.Sts2Run_GetStateList.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.Sts2Run_DebugSetHp.restype = ctypes.c_int
_lib.Sts2Run_DebugSetHp.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_DebugGainMaxHp.restype = ctypes.c_int
_lib.Sts2Run_DebugGainMaxHp.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_DebugEnterNextAct.restype = ctypes.c_int
_lib.Sts2Run_DebugEnterNextAct.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

_lib.Sts2Run_DebugUpgradeDeck.restype = ctypes.c_int
_lib.Sts2Run_DebugUpgradeDeck.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

_lib.Sts2Run_DebugResolveCombat.restype = ctypes.c_int
_lib.Sts2Run_DebugResolveCombat.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.Sts2Run_DebugSetPlayer.restype = ctypes.c_int
_lib.Sts2Run_DebugSetPlayer.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

try:  # run API 32; a library without it is refused by the version check unless STS2_ALLOW_STALE_NATIVE is set
    _lib.Sts2Run_DebugSetPlayerEx.restype = ctypes.c_int
    _lib.Sts2Run_DebugSetPlayerEx.argtypes = [
        *_lib.Sts2Run_DebugSetPlayer.argtypes[:11],  # handle .. max_potion_slots, as above
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    _HAS_DEBUG_SET_PLAYER_EX = True
except AttributeError:
    _HAS_DEBUG_SET_PLAYER_EX = False

_lib.Sts2Run_SetAnchorTables.restype = ctypes.c_int
_lib.Sts2Run_SetAnchorTables.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.Sts2Run_ReanchorFight.restype = ctypes.c_int
_lib.Sts2Run_ReanchorFight.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_lib.Sts2Run_ReanchorFightHidden.restype = ctypes.c_int
_lib.Sts2Run_ReanchorFightHidden.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_lib.Sts2Run_DebugSetStreams.restype = ctypes.c_int
_lib.Sts2Run_DebugSetStreams.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

_lib.Sts2Run_GetPhase.restype = ctypes.c_int
_lib.Sts2Run_GetPhase.argtypes = [ctypes.c_int]

_lib.Sts2_AiStateAssumed.restype = ctypes.c_int
_lib.Sts2_AiStateAssumed.argtypes = [ctypes.c_int]
_lib.Sts2Run_AiStateAssumed.restype = ctypes.c_int
_lib.Sts2Run_AiStateAssumed.argtypes = [ctypes.c_int]
_lib.Sts2Run_PlayerWon.restype = ctypes.c_int
_lib.Sts2Run_PlayerWon.argtypes = [ctypes.c_int]

_lib.Sts2Run_EncounterId.restype = ctypes.c_int
_lib.Sts2Run_EncounterId.argtypes = [ctypes.c_int]

_lib.Sts2Run_GetShuffleRngCallCount.restype = ctypes.c_int
_lib.Sts2Run_GetShuffleRngCallCount.argtypes = [ctypes.c_int]

_lib.Sts2Run_GetNicheRngCallCount.restype = ctypes.c_int
_lib.Sts2Run_GetNicheRngCallCount.argtypes = [ctypes.c_int]

_lib.Sts2Run_Destroy.restype = None
_lib.Sts2Run_Destroy.argtypes = [ctypes.c_int]

# ── public wrappers ───────────────────────────────────────────────────────────

OBS_SIZE: int = _lib.Sts2_ObsSize()
MAX_ENEMIES: int = _lib.Sts2_MaxEnemies()

# The level the plain resets deal (`reset`, and `reset_encounter` without a total_floor),
# and the max HP every run starts on. Both are the emulator's definitions, read rather
# than restated.
DEFAULT_ASCENSION: int = _lib.Sts2_DefaultAscension()
STARTING_MAX_HP: int = _lib.Sts2_StartingMaxHp()

# The combat observation's layout, read from the emulator rather than restated here.
# These used to be magic numbers in scripts/trace.py, which went silently wrong the
# moment a card slot grew from two fields to four: it decoded zero enemies and 77 live
# fixture tests failed at once, none of them about the observation.
OBS_CARD_SLOT_SIZE: int = _lib.Sts2_ObsCardSlotSize()
OBS_HAND_OFFSET: int = _lib.Sts2_ObsHandOffset()
OBS_MAX_HAND: int = _lib.Sts2_ObsMaxHand()
OBS_PLAYER_BUFF_OFFSET: int = _lib.Sts2_ObsPlayerBuffOffset()
OBS_MAX_PLAYER_BUFFS: int = _lib.Sts2_ObsMaxPlayerBuffs()
OBS_MAX_ENEMY_BUFFS: int = _lib.Sts2_ObsMaxEnemyBuffs()
OBS_ENEMY_OFFSET: int = _lib.Sts2_ObsEnemyOffset()
OBS_ENEMY_SLOT_SIZE: int = _lib.Sts2_ObsEnemySlotSize()
OBS_SECONDARY_INTENT_OFFSET: int = _lib.Sts2_ObsSecondaryIntentOffset()

# The open card-selection block: whether a screen is up, how many cards it offers, and
# the cards themselves. It is the one part of the observation that cannot be derived from
# the offsets above -- it sits between the secondary-intent block and the orbs, and
# neither of the two block sizes that separate them is published.
OBS_SELECTION_KIND_OFFSET: int = _lib.Sts2_ObsSelectionKindOffset()
OBS_SELECTION_COUNT_OFFSET: int = _lib.Sts2_ObsSelectionCountOffset()
OBS_SELECTION_OFFSET: int = _lib.Sts2_ObsSelectionOffset()
OBS_MAX_SELECTION_CANDIDATES: int = _lib.Sts2_ObsMaxSelectionCandidates()

# Defect's orb ring. Each slot is (type + 1, passive value, evoke value), with 0 in the
# first field for an empty slot -- Lightning is type 0, so a raw type could not be told
# from an empty one. The two values are what the orb is WORTH right now, Focus already in
# them, because a Dark orb's evoke is whatever it has banked and a Glass orb's is whatever
# it has left; neither is derivable from the type.
OBS_ORB_CAPACITY_OFFSET: int = _lib.Sts2_ObsOrbCapacityOffset()
OBS_ORB_OFFSET: int = _lib.Sts2_ObsOrbOffset()
OBS_INCOMING_OFFSET: int = _lib.Sts2_ObsIncomingOffset()
OBS_MAX_INCOMING_CARDS: int = _lib.Sts2_ObsMaxIncomingCards()
OBS_MAX_ORBS: int = _lib.Sts2_ObsMaxOrbs()
OBS_ORB_SLOT_SIZE: int = _lib.Sts2_ObsOrbSlotSize()
RUN_OBS_SIZE: int = _lib.Sts2Run_ObsSize()
RUN_MAX_ACTIONS: int = _lib.Sts2Run_MaxActions()
RUN_INFO_SIZE: int = _lib.Sts2Run_InfoSize()


def _run_obs_layout() -> dict[str, int]:
    """Where the run observation's deck and relic blocks sit, read from the native side.

    Hard-coding these on this side is how the run observation and its readers drifted
    apart before: the offsets move whenever a block grows, and a stale number reads the
    wrong column rather than failing.

    Raises:
        RuntimeError: if the native library reports a layout this build cannot read.

    """
    size = 13
    buf = (ctypes.c_int * size)()
    written = int(_lib.Sts2Run_ObsLayout(buf, size))
    if written != size:
        raise RuntimeError(
            f"Sts2Run_ObsLayout wrote {written} numbers, expected {size}. "
            "Rebuild the native library.",
        )
    keys = (
        "scalars",
        "deck_offset",
        "max_deck",
        "deck_slot_size",
        "relic_offset",
        "max_relics",
        "relic_slot_size",
        "shop_offset",
        "shop_slots",
        "shop_slot_size",
        "map_choices",
        "map_node_type_offset",
        # Lasting Candy's extra card-reward option: its id, then whether it is upgraded.
        # Appended to the observation AND to this layout, so neither renumbers.
        "fourth_reward_offset",
    )
    return dict(zip(keys, (int(value) for value in buf), strict=True))


RUN_OBS_LAYOUT: dict[str, int] = _run_obs_layout()


def create(seed: int) -> int:
    return _lib.Sts2_Create(seed)


def clone(
    handle: int,
    resample_hidden: bool,
    resample_seed: int,
    obs_buf: ctypes.Array,
) -> int:
    """Fork a combat into a new handle.

    With resample_hidden the copy is DETERMINIZED off resample_seed: every stream is
    reseeded, so the rolls still to come differ, and the draw pile is reshuffled, which
    a reseed alone cannot reach because that order was already materialised. Without it
    the copy is faithful, which for a tree search is an oracle -- see
    docs/agent-interface.md.

    Raises:
        RuntimeError: If the source handle is unknown or the handle pool is full.

    """
    handle_out = int(
        _lib.Sts2_Clone(handle, 1 if resample_hidden else 0, resample_seed, obs_buf),
    )
    if handle_out < 0:
        raise RuntimeError("Sts2_Clone failed: unknown handle or the pool is full.")
    return handle_out


class SnapshotError(Exception):
    """Raised when `Sts2_LoadCombatSnapshot` refuses an `sts2-emu-snapshot/1` document.

    The message is `Sts2_GetLastError`'s text: the JSON path of the field at fault, e.g.
    ``piles.hand[0].def_id: unknown card id 999999``. See
    docs/superpowers/specs/2026-09-24-snapshot-schemas.md Sec 2.
    """


class ExportRefused(SnapshotError):
    """Raised when `Sts2_GetStateJson` refuses to write a combat as a document.

    The combat is stopped on a screen raised in the middle of a step, and the step's
    replay from the document's public start would not reach that screen the way the step
    went, so the document would load as another position (CombatSnapshot.Write). The
    combat itself is untouched and can still be stepped. A `SnapshotError`, so code that
    handles a refused document handles this too.
    """


class StepRefused(RuntimeError):
    """Raised when `Sts2_Step` refuses an answer to a stopped step's screen.

    The answer replays the step from its start, and the replay went another way than the
    step did (another screen, other cards off the draw pile, answers left over): a loaded
    document's step in a world where it does not reach its screen. The combat is left on
    its screen as it was, and the observation buffer shows it. A `RuntimeError`, so code
    that handles a failed native call handles this too.
    """


def _get_last_error() -> str:
    """`Sts2_GetLastError`'s text (size-and-retry): why the last load, export or answer was refused."""
    capacity = 256
    while True:
        buf = (ctypes.c_ubyte * capacity)()
        length = int(_lib.Sts2_GetLastError(buf, capacity))
        if length <= capacity:
            return bytes(buf[:length]).decode("utf-8")
        capacity = length


_lib.Sts2_SetEngine.restype = ctypes.c_int
_lib.Sts2_SetEngine.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.Sts2_GetEngine.restype = ctypes.c_int
_lib.Sts2_GetEngine.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]


def set_engine(spec: str) -> str:
    """Make the handles created after this call on engine `spec` ("old", "core2", "core2:F08,F10"); returns its name.

    A handle keeps the engine it was made with.

    Raises:
        RuntimeError: For a spec the library refuses (the engine unchanged).

    """
    raw = spec.encode("utf-8")
    if int(_lib.Sts2_SetEngine(raw, len(raw))) != 0:
        raise RuntimeError(f"Sts2_SetEngine({spec!r}) refused: {_get_last_error()}")
    return engine()


def engine() -> str:
    """Return the engine new handles are made with, as the library names it: "old" or "core2:<families>"."""
    capacity = 64
    while True:
        buf = (ctypes.c_ubyte * capacity)()
        length = int(_lib.Sts2_GetEngine(buf, capacity))
        if length <= capacity:
            return bytes(buf[:length]).decode("utf-8")
        capacity = length


def _apply_engine_env() -> None:
    spec = os.environ.get(_ENGINE_ENV, "").strip()
    if spec and spec != "old":
        set_engine(spec)  # refused: raise at import, so no block starts on the wrong engine


_apply_engine_env()


def load_combat_snapshot(doc: dict | str | bytes, seed: int) -> int:
    """Build a combat from an `sts2-emu-snapshot/1` document.

    So search can start from a live game position instead of from a reset. `doc` is a
    dict (JSON-encoded here), a JSON string, or already-encoded UTF-8 bytes.
    `seed` seeds every stream and orders the unknown middle of the draw pile -- the same
    role it plays for `create`.

    Raises:
        SnapshotError: if the document is refused. The message names the field, per
            `Sts2_GetLastError`.
        RuntimeError: if the handle pool is full.

    """
    if isinstance(doc, dict):
        json_bytes = json.dumps(doc).encode("utf-8")
    elif isinstance(doc, str):
        json_bytes = doc.encode("utf-8")
    else:
        json_bytes = bytes(doc)

    buf = (ctypes.c_ubyte * len(json_bytes))(*json_bytes) if json_bytes else None
    handle = int(_lib.Sts2_LoadCombatSnapshot(buf, len(json_bytes), seed, None))
    if handle == -1:
        raise SnapshotError(_get_last_error())
    if handle == -2:
        raise RuntimeError("Sts2_LoadCombatSnapshot failed: the handle pool is full.")
    return handle


def reset(handle: int, obs_buf: ctypes.Array) -> None:
    _lib.Sts2_Reset(handle, obs_buf)


def reset_encounter(
    handle: int,
    encounter_id: int,
    obs_buf: ctypes.Array,
    completed_combat_rooms: int = -1,
    total_floor: int | None = None,
    ascension: int = DEFAULT_ASCENSION,
) -> None:
    """Reset into a chosen encounter.

    completed_combat_rooms is the weak context, read only for Corpse Slugs (9) and
    Seapunk (12), the weak ids their normal twins (88, 22) once shared: [0,3) keeps the
    weak fight, -1 (the default) deals the normal one, whichever reset runs. Every other
    id names its own fight (WeakContext.Resolve, the one rule at the C ABI).

    total_floor is the run's TotalFloor, and it is not optional decoration: it is the
    missing term in the per-encounter RNG seed (run seed + floor + hash of the encounter
    id) that Slimes rosters and Corpse Slug starting moves are rolled from. Leave it
    None and those encounters fall back to the combat rng, which does NOT match the
    live game.

    ascension is an input to enemy data, not a difficulty label: the game picks monster
    damage with GetValueIfAscension(level, high, low), so the same enemy hits for
    different amounts at A8 and A10. It also sets the player's start, by the run's own
    rules at that level: 80/80 HP below WearyTraveler (A2) and 64/80 from it, three
    potion slots below TightBelt (A4) and two from it, and Ascender's Bane in the deck
    only from AscendersBane (A5). It only reaches the emulator through the floor-aware
    reset: without a total_floor the fight is the default level's.

    Raises:
        ValueError: if an ascension other than DEFAULT_ASCENSION is asked for without a
            total_floor, since the reset that would run takes no ascension and the
            level would be dropped without a word. Pass total_floor (1 for a run's first
            combat) to honour it.

    """
    if total_floor is None and ascension != DEFAULT_ASCENSION:
        raise ValueError(
            f"reset_encounter: ascension {ascension} reaches the emulator only through "
            "the floor-aware reset. Pass total_floor (1 for a run's first combat); "
            f"without it the fight is A{DEFAULT_ASCENSION}'s.",
        )
    if total_floor is not None:
        _lib.Sts2_ResetEncounterAtFloor(
            handle,
            encounter_id,
            completed_combat_rooms,
            total_floor,
            ascension,
            obs_buf,
        )
    elif completed_combat_rooms == -1:
        _lib.Sts2_ResetEncounter(handle, encounter_id, obs_buf)
    else:
        _lib.Sts2_ResetEncounterWeak(
            handle,
            encounter_id,
            completed_combat_rooms,
            obs_buf,
        )


def debug_add_card_to_hand(
    handle: int,
    card_id: int,
    obs_buf: ctypes.Array,
    upgraded: bool = False,
) -> None:
    """Put a card on top of the hand, mirroring the mod's debug_add_card.

    For differential captures that must reach a state the starter deck cannot. The hand
    rather than the deck, so no shuffle has to agree between the two sides.
    """
    _lib.Sts2_DebugAddCardToHand(handle, card_id, 1 if upgraded else 0, obs_buf)


def pending_selection_kind(handle: int) -> int:
    """Report the kind of card selection the combat is waiting on, or 0 for none."""
    return int(_lib.Sts2_PendingSelectionKind(handle))


# The game's own stream names, in the order `Sts2_RngCallCount` expects. Keyed by the
# name the MOD reports under `battle.rng`, so the two sides can be zipped by name rather
# than by an index both files have to agree about.
RNG_STREAMS = {
    "Shuffle": 0,
    "CombatTargets": 1,
    "CombatCardSelection": 2,
    "CombatCardGeneration": 3,
    "CombatEnergyCosts": 4,
    "CombatOrbGeneration": 5,
    "Niche": 6,
    "MonsterAi": 7,
}


def rng_call_count(handle: int, stream: int) -> int:
    """Draws taken from one named stream, or -1 if this combat never built it."""
    return int(_lib.Sts2_RngCallCount(handle, stream))


def debug_gain_max_hp(handle: int, amount: int, obs_buf: ctypes.Array) -> None:
    """Raise max HP and heal by it, mirroring the mod's debug_gain_max_hp."""
    _lib.Sts2_DebugGainMaxHp(handle, amount, obs_buf)


def reset_encounter_with_extra_cards(
    handle: int,
    extra_card_ids: list[int],
    encounter_id: int,
    obs_buf: ctypes.Array,
    completed_combat_rooms: int = -1,
    total_floor: int = 0,
    ascension: int = 8,
) -> None:
    """reset_encounter with extra cards appended to the starter deck.

    For captures that must reach a state the starter deck cannot: a Phrog Parasite's
    Wrigglers only spawn when it dies. The live side adds the same cards with
    debug_add_card, and both sides append before shuffling. The starter deck, HP and belt
    are the level's, as in reset_encounter.
    """
    extra_buf = (ctypes.c_int * len(extra_card_ids))(*extra_card_ids)
    _lib.Sts2_ResetEncounterAtFloorWithExtraCards(
        handle,
        extra_buf,
        len(extra_card_ids),
        encounter_id,
        completed_combat_rooms,
        total_floor,
        ascension,
        obs_buf,
    )


def can_enchant(card_id: int, enchantment: int) -> bool:
    """Whether the enchantment may legally be applied to the card.

    ``reset_arena`` applies whatever it is handed without filtering, so this is the only
    thing standing between a sampler and an illegal position. Asked of the engine rather
    than restated here: the rule changes whenever an enchantment is added.
    """
    return bool(_lib.Sts2_CanEnchant(card_id, enchantment))


def starting_deck(ascension: int) -> list[int]:
    """The deck a run at `ascension` starts with, as the floor-aware resets deal it.

    Ascender's Bane is in it only from AscendersBane (A5). Asked of the engine rather than
    restated, so a caller building its own start (the arena) deals what a run deals.
    """
    capacity = 16
    while True:
        buf = (ctypes.c_int * capacity)()
        count = int(_lib.Sts2_StartingDeck(ascension, buf, capacity))
        if count <= capacity:
            return list(buf[:count])
        capacity = count


def starting_hp(ascension: int) -> int:
    """The HP a run at `ascension` starts on: STARTING_MAX_HP, 80% of it from WearyTraveler (A2)."""
    return int(_lib.Sts2_StartingHp(ascension))


def starting_potion_slots(ascension: int) -> int:
    """The potion slots a run at `ascension` starts with: three, two from TightBelt (A4)."""
    return int(_lib.Sts2_StartingPotionSlots(ascension))


def reset_arena(
    handle: int,
    deck_ids: list[int],
    obs_buf: ctypes.Array,
    enchantments: list[tuple[int, int]] | None = None,
    encounter_id: int = -1,
    relic_ids: list[int] | None = None,
    potion_ids: list[int] | None = None,
    player_hp: int = 64,
    player_max_hp: int = 80,
    player_gold: int = 0,
    ascension: int = 8,
    total_floor: int = 0,
    completed_combat_rooms: int = -1,
    max_potion_slots: int | None = None,
) -> None:
    """Start a combat from an arbitrary run position.

    Every other reset here starts from the fixed starter deck at full HP with no
    relics, which is one point in the space a deck-conditioned value function has to
    evaluate over. This one takes the whole position: deck, relics, potions, HP and
    ascension against a chosen encounter.

    A NEGATIVE card id means upgraded, matching CombatFactory's own
    ``new CardInstance(Math.Abs(id), id < 0)``.

    ``enchantments`` is a list of (enchantment_id, amount) running parallel to
    ``deck_ids``; pass None for a deck with none. Act 1 grants them by two routes, so
    this is not act-2 decoration: Self-Help Book, Stone of All Time and Symbiote are in
    both act-1 event pools (Sapphire Seed and Wood Carvings in Underdocks' as well), and
    six Shop-rarity relics grant one apiece.

    encounter_id of -1 rolls the seeded first-combat encounter instead of forcing one.

    ``max_potion_slots`` is the position's belt. None gives the belt a run at
    ``ascension`` starts with (``starting_potion_slots``); a position whose belt a relic
    widened passes its own. It is never narrower than the potions passed, so none is
    stranded where no action can reach it.
    """
    deck_buf = (ctypes.c_int * len(deck_ids))(*deck_ids)
    ench = list(enchantments or [])
    ench_ids = [e for e, _ in ench]
    ench_amts = [a for _, a in ench]
    ench_id_buf = (ctypes.c_int * max(1, len(ench_ids)))(*ench_ids)
    ench_amt_buf = (ctypes.c_int * max(1, len(ench_amts)))(*ench_amts)
    relics = list(relic_ids or [])
    relic_buf = (ctypes.c_int * max(1, len(relics)))(*relics)
    potions = list(potion_ids or [])
    potion_buf = (ctypes.c_int * max(1, len(potions)))(*potions)
    _lib.Sts2_ResetArena(
        handle,
        deck_buf,
        len(deck_ids),
        ench_id_buf,
        len(ench_ids),
        ench_amt_buf,
        len(ench_amts),
        encounter_id,
        relic_buf,
        len(relics),
        potion_buf,
        len(potions),
        player_hp,
        player_max_hp,
        player_gold,
        -1 if max_potion_slots is None else max_potion_slots,
        ascension,
        total_floor,
        completed_combat_rooms,
        obs_buf,
    )


def reset_with_deck(handle: int, deck_ids: list[int], obs_buf: ctypes.Array) -> None:
    deck_buf = (ctypes.c_int * len(deck_ids))(*deck_ids)
    _lib.Sts2_ResetWithDeck(handle, deck_buf, len(deck_ids), obs_buf)


def reset_with_deck_and_encounter(
    handle: int,
    deck_ids: list[int],
    encounter_id: int,
    obs_buf: ctypes.Array,
) -> None:
    deck_buf = (ctypes.c_int * len(deck_ids))(*deck_ids)
    _lib.Sts2_ResetWithDeckAndEncounter(
        handle,
        deck_buf,
        len(deck_ids),
        encounter_id,
        obs_buf,
    )


def reset_with_deck_encounter_and_relics(
    handle: int,
    deck_ids: list[int],
    encounter_id: int,
    relic_ids: list[int],
    obs_buf: ctypes.Array,
) -> None:
    deck_buf = (ctypes.c_int * len(deck_ids))(*deck_ids)
    relic_buf = (ctypes.c_int * len(relic_ids))(*relic_ids)
    _lib.Sts2_ResetWithDeckEncounterAndRelics(
        handle,
        deck_buf,
        len(deck_ids),
        encounter_id,
        relic_buf,
        len(relic_ids),
        obs_buf,
    )


def step(
    handle: int,
    action: int,
    obs_buf: ctypes.Array,
    reward_buf: ctypes.Array,
    target_enemy_index: int = -1,
) -> bool:
    """One action on a combat; True when the fight is over.

    Raises:
        StepRefused: if the action answers a screen raised in the middle of a step and the
            step's replay went another way than the step did (a loaded document in a world
            where it does not reach its screen). The combat stays on its screen as it was.

    """
    if target_enemy_index >= 0:
        terminal = _lib.Sts2_StepTargeted(
            handle,
            action,
            target_enemy_index,
            obs_buf,
            reward_buf,
        )
    else:
        terminal = _lib.Sts2_Step(handle, action, obs_buf, reward_buf)
    if terminal < 0:
        raise StepRefused(_get_last_error())
    return bool(terminal)


def valid_actions(handle: int, max_actions: int) -> ctypes.Array:
    buf = (ctypes.c_int * max_actions)()
    _lib.Sts2_ValidActions(handle, buf, max_actions)
    return buf


PILE_DRAW = 0
PILE_HAND = 1
PILE_DISCARD = 2
PILE_EXHAUST = 3

_PILE_NAMES = {
    "draw": PILE_DRAW,
    "hand": PILE_HAND,
    "discard": PILE_DISCARD,
    "exhaust": PILE_EXHAUST,
}


def get_pile(handle: int, pile: int | str = PILE_DRAW) -> list[tuple[int, bool]]:
    """Dump a combat pile in true order — index 0 is the top (next card drawn).

    Returns (card_def_id, upgraded) per card. The observation vector only carries
    pile counts, so this is the way to compare exact card sequences against the
    live game.

    Raises:
        ValueError: if the pile name is not one of the known combat piles.

    """
    pile_id = _PILE_NAMES[pile] if isinstance(pile, str) else pile
    if pile_id not in _PILE_NAMES.values():
        raise ValueError(
            f"unknown pile {pile!r}; expected one of {sorted(_PILE_NAMES)}",
        )

    capacity = 64
    while True:
        buf = (ctypes.c_int * (capacity * 2))()
        count = _lib.Sts2_GetPile(handle, pile_id, buf, capacity)
        if count < 0:
            raise ValueError(f"native rejected pile id {pile_id}")
        if count <= capacity:
            return [(buf[i * 2], bool(buf[i * 2 + 1])) for i in range(count)]
        capacity = count


def get_enemies(handle: int) -> list[tuple[int, int, int]]:
    """The living enemy roster as (def_id, slot, hp) triples, in roster order.

    The observation carries no def id -- an agent sees HP, block, intent and buffs, not
    WHICH creature it faces. That is right for learning and useless for a differential
    sweep, where a roster differing by one is a list of numbers until the creatures can be
    named.
    """
    capacity = 16
    while True:
        buf = (ctypes.c_int * (capacity * 3))()
        count = _lib.Sts2_GetEnemies(handle, buf, capacity)
        if count <= capacity:
            return [(buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]) for i in range(count)]
        capacity = count


def get_relics(handle: int) -> list[tuple[int, int]]:
    """The player's relics as (def id, counter) pairs, in the order the run holds them.

    The combat observation carries no relic block, so this is the only way to see what a
    combat started through ``reset_arena`` is actually holding.
    """
    capacity = 32
    while True:
        buf = (ctypes.c_int * (capacity * 2))()
        count = _lib.Sts2_GetRelics(handle, buf, capacity)
        if count <= capacity:
            return [(buf[i * 2], buf[i * 2 + 1]) for i in range(count)]
        capacity = count


def get_executed(handle: int) -> list[int]:
    """Def ids whose intents executed during the enemy turn just past, in order."""
    capacity = 16
    while True:
        buf = (ctypes.c_int * capacity)()
        count = _lib.Sts2_GetExecuted(handle, buf, capacity)
        if count <= capacity:
            return [buf[i] for i in range(count)]
        capacity = count


def get_state_json(handle: int) -> dict:
    """Export a combat as an `sts2-emu-snapshot/1` document (size-and-retry).

    The counterpart to `load_combat_snapshot`: `get_state_json(load_combat_snapshot(doc,
    seed))` round-trips `doc`, modulo the unordered draw-pile middle, which is a multiset
    (docs/superpowers/specs/2026-09-24-snapshot-schemas.md Sec 2.1).

    The position as the player sees it. A combat stopped on a screen raised in the middle
    of a step is written with the public part of that step (`stopped_step`: the kind of
    step, the card or potion in play, the answers so far, the step's start as the player
    saw it; never the true draw order or an RNG stream): loaded, answering the screen
    finishes the step as answering in memory does (CombatSnapshot.cs). The block also
    carries what the player saw the step do before its screen (`taken`, `screens`, and
    `peeks`: whether I Am Invincible played itself off the top), which the load holds the
    replay to.

    Raises:
        ExportRefused: if the combat is stopped in the middle of a step whose replay from
            the document's start would not reach its screen: no document is written rather
            than one that loads as another position. The message is `Sts2_GetLastError`'s.
        RuntimeError: if the handle is unknown.

    """
    return _json_export(_lib.Sts2_GetStateJson, "Sts2_GetStateJson", handle)


def get_save_json(handle: int) -> dict:
    """Export a combat as the emulator's save document (size-and-retry).

    `get_state_json`'s document plus the true world of a step stopped on a screen
    (`stopped_step`'s `before_draw`, `before_streams`, `answer_indices` and `hidden`): the
    step's start with its TRUE draw order and the exact state of every RNG stream, so that
    `load_combat_snapshot` of it replays the step from its true start. Banned outside verification and
    save code, as `get_pile` and `rng_call_count` are (review condition E12): search and
    agent code that read it read the future. An unknown handle raises RuntimeError.
    """
    return _json_export(_lib.Sts2_GetSaveJson, "Sts2_GetSaveJson", handle)


def _json_export(export: Callable[..., int], name: str, handle: int) -> dict:
    """Read a size-and-retry JSON export of one combat.

    Raises:
        RuntimeError: if the handle is unknown.

    """
    capacity = 4096
    while True:
        buf = (ctypes.c_ubyte * capacity)()
        length = int(export(handle, buf, capacity))
        if length == -3:
            raise ExportRefused(_get_last_error())
        if length < 0:
            raise RuntimeError(f"{name} failed: unknown handle {handle}.")
        if length <= capacity:
            return json.loads(bytes(buf[:length]))
        capacity = length


def player_won(handle: int) -> bool:
    return bool(_lib.Sts2_PlayerWon(handle))


def encounter_id(handle: int) -> int:
    return int(_lib.Sts2_EncounterId(handle))


def destroy(handle: int) -> None:
    _lib.Sts2_Destroy(handle)


def run_create() -> int:
    return int(_lib.Sts2Run_Create())


def run_clone(
    handle: int,
    resample_hidden: bool,
    resample_seed: int,
    obs_buf: ctypes.Array,
) -> int:
    """Fork a run into a new handle.

    With resample_hidden, everything the agent has not been shown is resampled off
    resample_seed: future rewards, shop stock, encounter composition, and the unseen
    part of the draw pile. Without it the copy is faithful, which for a tree search
    is an oracle -- see docs/agent-interface.md.

    Raises:
        RuntimeError: If the source handle is unknown or the handle pool is full, or (-3) the library refused the clone; the
            message is `Sts2_GetLastError`'s. Under `STS2_ENGINE=core2` the copy carries the run's kernel.

    """
    handle_out = int(
        _lib.Sts2Run_Clone(handle, 1 if resample_hidden else 0, resample_seed, obs_buf),
    )
    if handle_out == -3:
        raise RuntimeError(f"Sts2Run_Clone refused: {_get_last_error()}")
    if handle_out < 0:
        raise RuntimeError("Sts2Run_Clone failed: unknown handle or the pool is full.")
    return handle_out


def run_fork_combat(
    handle: int,
    resample_hidden: bool,
    resample_seed: int,
    obs_buf: ctypes.Array,
) -> int:
    """Hand the fight a run stands in to a new COMBAT handle (rl/run_train/DESIGN.md 6.5).

    With resample_hidden the fight is exactly the one `run_clone` with the same seed
    holds: every stream rebuilt off resample_seed at the same call counts, and the unseen
    draw pile reshuffled. Without it the copy is faithful, an oracle. The run is not
    touched; the new handle is an ordinary combat handle, freed with `destroy`, and
    obs_buf receives its combat observation.

    Raises:
        RuntimeError: If the run is unknown or not in a fight, or the combat pool is full.

    """
    handle_out = int(
        _lib.Sts2Run_ForkCombat(
            handle,
            1 if resample_hidden else 0,
            resample_seed,
            obs_buf,
        ),
    )
    if handle_out == -2:
        raise RuntimeError("Sts2Run_ForkCombat failed: the combat handle pool is full.")
    if handle_out == -3:
        raise RuntimeError(f"Sts2Run_ForkCombat refused: {_get_last_error()}")
    if handle_out < 0:
        raise RuntimeError("Sts2Run_ForkCombat failed: unknown run, or not in a fight.")
    return handle_out


def run_reset(
    handle: int,
    seed: str,
    obs_buf: ctypes.Array,
    ascension: int = 0,
) -> int:
    """Start a new run on this handle at the given ascension level.

    Every ascension rule in the run and its combats reads the level -- starting HP, the
    elite count, gold, the belt, Ascender's Bane, card odds, monster HP and damage. The
    default is 0; the committed traces and live fixtures were captured at 8 and pass it.

    Returns:
        0 on success, -1 for an unknown handle, -2 for a level the run engine does not
        model (anything outside the game's 0-10).

    """
    seed_bytes = seed.encode("utf-8")
    seed_buf = (ctypes.c_ubyte * len(seed_bytes))(*seed_bytes)
    return int(
        _lib.Sts2Run_Reset(
            handle,
            seed_buf,
            len(seed_bytes),
            ascension,
            obs_buf,
        ),
    )


# The five characters the game offers, in the emulator's `PlayerCharacter` order (the index Sts2Run_ResetAs takes).
RUN_CHARACTERS: tuple[str, ...] = ("IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT")

# Away from a fight a held potion is used with `RUN_USE_POTION_ACTION + slot` and thrown away with
# `RUN_DISCARD_POTION_ACTION + slot` (slot 0-5), on any screen whose action mask lists them (`RunPotionUse`).
RUN_USE_POTION_ACTION: int = 242
RUN_DISCARD_POTION_ACTION: int = 248
RUN_POTION_SLOTS: int = 6

# The run action mask has versions (D32), a setting of the run handle. Version 1 is the default and the mask as it was;
# version 2 also lists the discard actions on the card reward, deck-select, bundle-select and Crystal Sphere screens;
# version 3 (D34) also lists the use actions of the AnyTime potions there (Blood Potion, Fruit Juice, Entropic Brew);
# version 4 (D34) is the same mask as 3 and is the setting under which the run layer shows Welcome to Wongos' Leave (action 3).
# `run_set_mask_version` picks one; `Sts2RunEnv(mask_version=...)` does it per env.
RUN_MASK_VERSION_DEFAULT: int = 1
RUN_MASK_VERSION_OPEN_DISCARD: int = 2
RUN_MASK_VERSION_OPEN_USE: int = 3
RUN_MASK_VERSION_WONGOS_LEAVE: int = 4
RUN_MASK_VERSIONS: tuple[int, ...] = (1, 2, 3, 4)


def run_character_index(character: str) -> int:
    """The `Sts2Run_ResetAs` index of a character id ("SILENT", or "CHARACTER.SILENT"; any case).

    Raises:
        ValueError: for a character the game does not offer.

    """
    name = str(character).upper().removeprefix("CHARACTER.")
    if name not in RUN_CHARACTERS:
        raise ValueError(f"unknown character {character!r}; the game offers {', '.join(RUN_CHARACTERS)}")
    return RUN_CHARACTERS.index(name)


def run_reset_character(
    handle: int,
    seed: str,
    obs_buf: ctypes.Array,
    ascension: int = 0,
    character: str = "IRONCLAD",
) -> int:
    """Start a new run on this handle as `character` (its start deck, relic, max HP and pools) at `ascension`.

    Returns:
        0 on success, -1 for an unknown handle, -2 for an ascension the run engine does not model.

    Raises:
        ValueError: for a character the game does not offer.

    """
    index = run_character_index(character)
    seed_bytes = seed.encode("utf-8")
    seed_buf = (ctypes.c_ubyte * len(seed_bytes))(*seed_bytes)
    return int(
        _lib.Sts2Run_ResetAs(
            handle,
            seed_buf,
            len(seed_bytes),
            ascension,
            index,
            obs_buf,
        ),
    )


def run_set_mask_version(handle: int, version: int) -> None:
    """Pick the version of the action mask this run handle writes (`RUN_MASK_VERSIONS`); `run_reset*` keeps it.

    Raises:
        ValueError: for an unknown handle or a version the mask does not have.

    """
    status = int(_lib.Sts2Run_SetMaskVersion(handle, int(version)))
    if status != 0:
        raise ValueError(f"Sts2Run_SetMaskVersion({handle}, {version}) failed with status {status}")


def run_potion_action_reason(handle: int, action: int) -> str:
    """Why the use or discard potion `action` is refused on this run right now; "" when it is legal.

    Raises:
        ValueError: for an unknown handle or an action that is not a potion action.

    """
    buf = (ctypes.c_ubyte * 256)()
    n = int(_lib.Sts2Run_PotionActionReason(handle, action, buf, len(buf)))
    if n < 0:
        raise ValueError(f"Sts2Run_PotionActionReason({handle}, {action}) failed with status {n}")
    return bytes(buf[:n]).decode("utf-8")


def run_step(
    handle: int,
    action: int,
    target_enemy_index: int,
    obs_buf: ctypes.Array,
    reward_buf: ctypes.Array,
    terminal_buf: ctypes.Array,
    truncated_buf: ctypes.Array,
) -> int:
    return int(
        _lib.Sts2Run_Step(
            handle,
            action,
            target_enemy_index,
            obs_buf,
            reward_buf,
            terminal_buf,
            truncated_buf,
        ),
    )


def run_start_combat(
    handle: int,
    deck_ids: list[int],
    encounter_id: int,
    relic_ids: list[int],
    player_hp: int,
    player_max_hp: int,
    potion_ids: list[int],
    player_gold: int,
    completed_combat_rooms_before_current: int,
    obs_buf: ctypes.Array,
) -> int:
    deck_buf = (ctypes.c_int * len(deck_ids))(*deck_ids)
    relic_buf = (ctypes.c_int * len(relic_ids))(*relic_ids)
    potion_buf = (ctypes.c_int * len(potion_ids))(*potion_ids)
    return int(
        _lib.Sts2Run_StartCombat(
            handle,
            deck_buf,
            len(deck_ids),
            encounter_id,
            relic_buf,
            len(relic_ids),
            player_hp,
            player_max_hp,
            potion_buf,
            len(potion_ids),
            player_gold,
            completed_combat_rooms_before_current,
            obs_buf,
        ),
    )


def run_action_mask(handle: int, max_actions: int) -> ctypes.Array:
    buf = (ctypes.c_int * max_actions)()
    status = int(_lib.Sts2Run_ActionMask(handle, buf, max_actions))
    if status != 0:
        raise RuntimeError(f"Sts2Run_ActionMask failed with status {status}.")
    return buf


def run_info(handle: int) -> ctypes.Array:
    buf = (ctypes.c_int * RUN_INFO_SIZE)()
    status = int(_lib.Sts2Run_GetInfo(handle, buf, RUN_INFO_SIZE))
    if status != 0:
        raise RuntimeError(f"Sts2Run_GetInfo failed with status {status}.")
    return buf


# Run-generation lists the player has not been shown: the act's normal and elite encounter
# sequences and its event sequence (what is behind each map node, before the node is entered).
# DESIGN.md 1.1 and 7.6 L4: no agent-visible readout, feature or prior reads them. They stay in
# the engine, which resolves a node from them, and a generation test may ask for them.
HIDDEN_RUN_LISTS = frozenset({11, 12, 13})


def run_state_list(
    handle: int, list_id: int, capacity: int = 256, *, hidden: bool = False
) -> tuple[int, ...]:
    """One of the run's state lists. The hidden generation lists (`HIDDEN_RUN_LISTS`) are refused
    unless the caller passes `hidden=True`, which only a test of run generation does."""
    if list_id in HIDDEN_RUN_LISTS and not hidden:
        raise PermissionError(
            f"run state list {list_id} holds what the player has not been shown "
            "(encounters and events still to come); pass hidden=True only from a generation test"
        )
    buf = (ctypes.c_int * capacity)()
    count = int(_lib.Sts2Run_GetStateList(handle, list_id, buf, capacity))
    if count < 0:
        raise RuntimeError(
            f"Sts2Run_GetStateList failed with status {count} for list {list_id}.",
        )
    return tuple(int(buf[i]) for i in range(min(count, capacity)))


def run_phase(handle: int) -> int:
    return int(_lib.Sts2Run_GetPhase(handle))


def ai_state_assumed(handle: int) -> bool:
    """True when this combat fork kept the true hidden AI state (DESIGN.md 1.1)."""
    return bool(_lib.Sts2_AiStateAssumed(handle))


def run_ai_state_assumed(handle: int) -> bool:
    """True when the run's fight is a fork that kept the true hidden AI state."""
    return bool(_lib.Sts2Run_AiStateAssumed(handle))


def run_player_won(handle: int) -> bool:
    return bool(_lib.Sts2Run_PlayerWon(handle))


def run_encounter_id(handle: int) -> int:
    return int(_lib.Sts2Run_EncounterId(handle))


def run_get_shuffle_rng_call_count(handle: int) -> int:
    return int(_lib.Sts2Run_GetShuffleRngCallCount(handle))


def run_get_niche_rng_call_count(handle: int) -> int:
    return int(_lib.Sts2Run_GetNicheRngCallCount(handle))


def run_debug_set_hp(
    handle: int,
    hp: int,
    max_hp: int,
    obs_buf: ctypes.Array,
) -> None:
    """Soak-only: hand a run extra HP so it can reach the act's boss.

    Raises:
        RuntimeError: if the handle is not a live run.

    """
    if _lib.Sts2Run_DebugSetHp(handle, hp, max_hp, obs_buf) != 0:
        raise RuntimeError("Sts2Run_DebugSetHp failed")


def run_debug_gain_max_hp(handle: int, amount: int, obs_buf: ctypes.Array) -> None:
    """Mirror the mod's debug_gain_max_hp: raise the maximum AND heal by the amount.

    This is the one to use when replaying a BUFFED capture. run_debug_set_hp sets
    absolutes and does not heal, so a replay built on it diverges on HP one step after
    the buff.

    Raises:
        RuntimeError: if the handle is not a live run.

    """
    if _lib.Sts2Run_DebugGainMaxHp(handle, amount, obs_buf) != 0:
        raise RuntimeError("Sts2Run_DebugGainMaxHp failed")


def run_debug_enter_next_act(handle: int, obs_buf: ctypes.Array) -> bool:
    """Enter the next act, as the mod's debug_enter_next_act does.

    Returns False when the run is already in its last act. This runs the same transition
    the boss reward does — what it skips is having to win act 1 to get there.

    Raises:
        RuntimeError: if the handle is not a live run.

    """
    result = int(_lib.Sts2Run_DebugEnterNextAct(handle, obs_buf))
    if result < 0:
        raise RuntimeError("Sts2Run_DebugEnterNextAct failed")
    return result == 1


def run_debug_upgrade_deck(handle: int, obs_buf: ctypes.Array) -> None:
    """Soak-only: upgrade every upgradable card in the deck.

    Raises:
        RuntimeError: if the handle is not a live run.

    """
    if _lib.Sts2Run_DebugUpgradeDeck(handle, obs_buf) != 0:
        raise RuntimeError("Sts2Run_DebugUpgradeDeck failed")


def run_debug_resolve_combat(
    handle: int,
    obs_buf: ctypes.Array,
    *,
    stolen_back_gold: int = 0,
    fat_gremlin_escaped: bool = False,
    merc_gold_stolen: bool = False,
) -> bool:
    """Replay-only: win the fight in progress without playing it.

    The run layer of a real run is replayed through this (scripts/replay_real_run.py);
    its fights are measured separately. The keywords are the fight's results its rewards
    read: gold a Heist hands back, and how a Gremlin Merc fight ended. Returns False when
    no fight is in progress.

    Raises:
        RuntimeError: if the handle is not a live run.

    """
    result = int(
        _lib.Sts2Run_DebugResolveCombat(
            handle,
            int(stolen_back_gold),
            int(bool(fat_gremlin_escaped)),
            int(bool(merc_gold_stolen)),
            obs_buf,
        ),
    )
    if result < -1:
        raise RuntimeError("Sts2Run_DebugResolveCombat failed")
    return result == 0


def _int_array(values):
    if values is None:
        return None, -1
    array = (ctypes.c_int * max(1, len(values)))(*values)
    return array, len(values)


def run_debug_set_player(
    handle: int,
    obs_buf: ctypes.Array,
    *,
    hp: int = -1,
    max_hp: int = -1,
    gold: int = -1,
    cards=None,
    relics=None,
    potions=None,
    max_potion_slots: int = -1,
    card_tinker=None,
    relic_counters=None,
) -> None:
    """Replay-only: set the player's holdings where a real run had them.

    ``cards`` is ``[(id, upgraded, enchantment, amount), ...]``; ``relics`` and
    ``potions`` are ids (a potion list is by slot, 0 for empty). ``None`` or a negative
    scalar leaves that holding alone. No pickup effect runs.

    Run API 32: ``card_tinker`` is ``[(type, rider), ...]`` per card in the order of ``cards``,
    Mad Science's Tinker Time type and rider as the GAME's enum values (-1 for no type);
    ``relic_counters`` is ``[(relic id, counter), ...]`` for held relics, put on the run's copy
    and the open fight's. With neither, the original export is called, as before.

    Raises:
        RuntimeError: if the handle is not a live run.

    """
    flat = None if cards is None else [int(v) for card in cards for v in card]
    card_buf, card_len = _int_array(flat)
    relic_buf, relic_len = _int_array(None if relics is None else list(relics))
    potion_buf, potion_len = _int_array(None if potions is None else list(potions))
    if card_tinker is None and relic_counters is None:
        status = _lib.Sts2Run_DebugSetPlayer(
            handle,
            int(hp),
            int(max_hp),
            int(gold),
            card_buf,
            card_len,
            relic_buf,
            relic_len,
            potion_buf,
            potion_len,
            int(max_potion_slots),
            obs_buf,
        )
    else:
        if not _HAS_DEBUG_SET_PLAYER_EX:
            raise RuntimeError("the native library has no Sts2Run_DebugSetPlayerEx (run API 32)")
        tinker_buf, tinker_len = _int_array(None if card_tinker is None else [int(v) for pair in card_tinker for v in pair])
        counter_buf, counter_len = _int_array(
            None if relic_counters is None else [int(v) for pair in relic_counters for v in pair]
        )
        status = _lib.Sts2Run_DebugSetPlayerEx(
            handle,
            int(hp),
            int(max_hp),
            int(gold),
            card_buf,
            card_len,
            relic_buf,
            relic_len,
            potion_buf,
            potion_len,
            int(max_potion_slots),
            tinker_buf,
            tinker_len,
            counter_buf,
            counter_len,
            obs_buf,
        )
    if status != 0:
        raise RuntimeError("Sts2Run_DebugSetPlayer failed")


_anchor_tables_set = False


def run_set_anchor_tables(id_map: dict | None = None, aliases: dict | None = None) -> None:
    """Hand the library the game's id tables its fight loader reads a record through (it reads no file).

    Once per process is enough; `run_reanchor_fight` does it on first use. By default the tables are the
    repo's ``data/id_map.json`` and ``tools/fidelity/power_aliases.json``.

    Raises:
        RuntimeError: if the library refuses the tables.

    """
    global _anchor_tables_set  # noqa: PLW0603
    root = _repo_root()
    if id_map is None:
        id_map = json.loads((root / "data" / "id_map.json").read_text(encoding="utf-8"))
    if aliases is None:
        aliases = json.loads((root / "tools" / "fidelity" / "power_aliases.json").read_text(encoding="utf-8"))
    text = json.dumps({"id_map": id_map, "aliases": aliases}).encode("utf-8")
    if _lib.Sts2Run_SetAnchorTables(text, len(text)) != 0:
        raise RuntimeError("Sts2Run_SetAnchorTables failed")
    _anchor_tables_set = True


def run_reanchor_fight(handle: int, record: dict | str | bytes, obs_buf: ctypes.Array) -> tuple[int, str]:
    """Put the fight the run stands in on the referee's record of it (a combat snapshot as an agent sees it).

    The record is written over a copy of the fight and swapped in only when all of it could be loaded, so a
    refused record leaves the run as it was. The combat's own streams and, where its cards are the record's, its
    draw order stay (the record carries neither). Returns ``(status, message)``: ``status`` is the number of
    fields set (0 or more, the message names them and how the enemies' next moves were set), or negative:
    -1 no fight (or no such run), -2 the record is not a JSON object, -3 no tables handed over, -4 the record
    names something the emulator has no id for, or an enemy it has no counterpart of (the message says which),
    -5 another failure. ``obs_buf`` is refreshed on success.
    """
    if not _anchor_tables_set:
        run_set_anchor_tables()
    if isinstance(record, dict):
        record = json.dumps(record)
    data = record.encode("utf-8") if isinstance(record, str) else bytes(record)
    message = ctypes.create_string_buffer(512)
    status = _lib.Sts2Run_ReanchorFight(handle, data, len(data), message, len(message), obs_buf)
    return status, message.value.decode("utf-8", "replace")


ANCHOR_STREAMS = 1
ANCHOR_DRAW_ORDER = 2


def run_reanchor_fight_hidden(
    handle: int, record: dict | str | bytes, obs_buf: ctypes.Array, flags: int = ANCHOR_STREAMS | ANCHOR_DRAW_ORDER
) -> tuple[int, str]:
    """`run_reanchor_fight` with the game's hidden state: the record carries ``"rng"`` (the game's run streams,
    ``{name: {"seed", "counter"}}``) and ``"draw"`` (the draw pile top first, in the game's order), and ``flags`` says
    which of the two the library reads (ANCHOR_STREAMS, ANCHOR_DRAW_ORDER). The combat's streams and the run's own
    are put on the game's counters; the draw pile is the game's order. Same status codes as `run_reanchor_fight`.

    For the lockstep's emulator side only: the record is the referee's hidden state and must never reach a search or
    an agent's observation. A search forks the position with ``resample_hidden=True``, which redraws all of it.
    """
    if not _anchor_tables_set:
        run_set_anchor_tables()
    if isinstance(record, dict):
        record = json.dumps(record)
    data = record.encode("utf-8") if isinstance(record, str) else bytes(record)
    message = ctypes.create_string_buffer(512)
    status = _lib.Sts2Run_ReanchorFightHidden(handle, data, len(data), int(flags), message, len(message), obs_buf)
    return status, message.value.decode("utf-8", "replace")


def run_debug_set_streams(handle: int, streams: dict | str | bytes) -> int:
    """Put the run's streams on the game's counters: ``{name: {"seed", "counter"}}`` as the referee driver reports a
    stream set (the run's streams and the player's). A stream with another seed, or a name the run lacks, is left
    alone; one ahead of the game's counter is moved back. Returns how many streams moved, or -1 no such run, -2 the
    text is not that object. Lockstep emulator side only (see `run_reanchor_fight_hidden`).
    """
    if isinstance(streams, dict):
        streams = json.dumps(streams)
    data = streams.encode("utf-8") if isinstance(streams, str) else bytes(streams)
    return int(_lib.Sts2Run_DebugSetStreams(handle, data, len(data)))


def run_destroy(handle: int) -> None:
    _lib.Sts2Run_Destroy(handle)


# ── the fight solver (docs/rebuild/FIGHT_SOLVER.md sections 1 and 4a, piece S4) ────────────────────────────────────────
#
# Two outputs, kept apart: solve_expected returns an ExpectedSolve, the only output a label writer, the regret benchmark
# or a grade may read; solve_hindsight returns a HindsightBound, a separate type sharing no field with it, for lost-fight
# attribution only. The gate is the library's: Sts2_SolveHindsight refuses unless the solver was made with
# allow_hindsight=True, on a handle whose streams are not the record's, and on a handle with an unset stream
# (hindsight-unset-streams). Binding the export lazily adds nothing (ctypes resolves an export on attribute access);
# tests/python/test_solver_hindsight_pins.py keeps the lab and the search from naming it. A refused result carries no
# number: its values are None here (NaN in the native structs). The ctypes structures mirror Interop/SolverExports.cs field
# for field (tests/python/test_solver_binding.py reads the C# to hold them alike). One solver handle serves one thread at a
# time; open_solver() destroys its handle on exit.

PRUNE_TRANSPOSITION = 1
PRUNE_MULTISET_HAND = 2
PRUNE_INTERCHANGEABLE_ENEMIES = 4
PRUNE_SUBTREE_REUSE = 8
PRUNE_DRAW_BY_VALUE = 16
PRUNE_MACRO_DRAW = 32
PRUNE_UTILITY_BOUND = 64
PRUNE_LETHAL_CLOSURE = 128
PRUNE_BRANCH_BOUND = 256
PRUNE_CLOSER_FIRST = 512
PRUNE_ALL = 1023
PRUNE_NO_BOUNDS = 63  # every pruning but the bounds round's (the engine round's PRUNE_ALL)
SOLVE_EXACT, SOLVE_BUDGET, SOLVE_ERROR, SOLVE_REFUSED = 0, 1, -1, -3
SOLVER_Q_STRIDE = 7  # 1 + the observation's enemy slots: an untargeted move, then one entry per target
_HINDSIGHT_LINE_CAP = 64
_SOLVER_MAX_ACTIONS = 64


class _SolveOptionsC(ctypes.Structure):
    _fields_ = [
        ("node_budget", ctypes.c_int64),
        ("step_budget", ctypes.c_int64),
        ("time_budget_ms", ctypes.c_int32),
        ("max_horizon", ctypes.c_int32),
        ("prune", ctypes.c_int32),
        ("chance_width_cap", ctypes.c_int32),
        ("forbidden_potions", ctypes.c_int32),
        ("utility", ctypes.c_int32),
        ("allow_hindsight", ctypes.c_int32),
        ("use_estimate", ctypes.c_int32),
    ]


class _ExpectedResultC(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("horizon", ctypes.c_int32),
        ("exact", ctypes.c_int32),
        ("move_count", ctypes.c_int32),
        ("best_action", ctypes.c_int32),
        ("best_target", ctypes.c_int32),
        ("q_stride", ctypes.c_int32),
        ("utility_kind", ctypes.c_int32),
        ("value", ctypes.c_double),
        ("exact_mass", ctypes.c_double),
        ("seconds", ctypes.c_double),
        ("nodes", ctypes.c_int64),
        ("steps", ctypes.c_int64),
        ("table_hits", ctypes.c_int64),
        ("leaves", ctypes.c_int64),
        ("horizon_leaves", ctypes.c_int64),
        ("allowed_potions", ctypes.c_int32),
    ]


class _HindsightOptionsC(ctypes.Structure):
    _fields_ = [
        ("search_node_budget", ctypes.c_int64),
        ("search_step_budget", ctypes.c_int64),
        ("search_time_ms", ctypes.c_int32),
        ("max_turns", ctypes.c_int32),
    ]


class _HindsightResultC(ctypes.Structure):
    _fields_ = [
        ("outcome", ctypes.c_int32),
        ("win_line_exists", ctypes.c_int32),
        ("complete", ctypes.c_int32),
        ("line_length", ctypes.c_int32),
        ("best_utility", ctypes.c_double),
        ("elapsed", ctypes.c_double),
        ("nodes_searched", ctypes.c_int64),
        ("engine_steps", ctypes.c_int64),
        ("line", ctypes.c_int32 * (2 * _HINDSIGHT_LINE_CAP)),
    ]


class _CurveEntryC(ctypes.Structure):
    _fields_ = [
        ("subset", ctypes.c_int32),
        ("curve_status", ctypes.c_int32),
        ("curve_exact", ctypes.c_int32),
        ("curve_horizon", ctypes.c_int32),
        ("expected_l0", ctypes.c_double),
        ("win_chance", ctypes.c_double),
        ("expected_win_hp", ctypes.c_double),
        ("curve_exact_mass", ctypes.c_double),
        ("held", ctypes.c_double * 8),
        ("allowed_potions", ctypes.c_int32),
    ]


class SolveOptions:
    """A solve's options (Solver/Expectimax.cs SolveOptions); a budget of 0 takes the library's default.

    allowed_potions is a slot mask (bit i: the potion in slot i may be used; -1 every slot). The native struct carries its
    complement (forbidden_potions), so a zeroed struct from any C caller leaves every potion usable.
    """

    __slots__ = ("node_budget", "step_budget", "time_budget_ms", "max_horizon", "prune", "chance_width_cap", "allowed_potions",
                 "use_estimate")

    def __init__(self, node_budget: int = 200_000, step_budget: int = 2_000_000, time_budget_ms: int = 0, max_horizon: int = 8,
                 prune: int = PRUNE_ALL, chance_width_cap: int = 32, allowed_potions: int = -1, use_estimate: bool = True):
        self.node_budget, self.step_budget, self.time_budget_ms = node_budget, step_budget, time_budget_ms
        self.max_horizon, self.prune, self.chance_width_cap = max_horizon, prune, chance_width_cap
        self.allowed_potions, self.use_estimate = allowed_potions, use_estimate

    def _c(self, allow_hindsight: bool = False) -> _SolveOptionsC:
        forbidden = ctypes.c_int32(~int(self.allowed_potions)).value
        return _SolveOptionsC(self.node_budget, self.step_budget, self.time_budget_ms, self.max_horizon, self.prune,
                              self.chance_width_cap, forbidden, 0, 1 if allow_hindsight else 0, 1 if self.use_estimate else 0)


class ExpectedSolve:
    """The expected value of a position (Sts2_SolveExpected): the only solver output labels, regret and grades may read.

    q maps (action, target) to the move's value, target -1 for none; a move with no value is left out. A refused solve
    (status -3) has no value: value and exact_mass are None, never a 0 that reads as a certain loss. allowed_potions is the
    potion slot mask the solve used (-1: every slot).
    """

    __slots__ = ("status", "reason", "horizon", "value", "exact", "exact_mass", "move_count", "best_action", "best_target", "q",
                 "nodes", "steps", "table_hits", "leaves", "horizon_leaves", "seconds", "allowed_potions")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def __repr__(self) -> str:
        return f"ExpectedSolve(status={self.status}, value={self.value}, exact={self.exact})"


class HindsightBound:
    """The hindsight bound (Sts2_SolveHindsight): an upper bound for lost-fight attribution, never a label, regret or grade.

    best_utility is the bound only when the search was complete (outcome 0); it is None when the search was cut (its best
    line is a lower bound on the bound, not the bound) or refused.
    """

    __slots__ = ("outcome", "refusal", "win_line_exists", "best_utility", "complete", "line", "nodes_searched", "engine_steps",
                 "elapsed")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def __repr__(self) -> str:
        return f"HindsightBound(win_line_exists={self.win_line_exists}, best_utility={self.best_utility})"


_lib.Sts2Solver_Create.restype = ctypes.c_int
_lib.Sts2Solver_Create.argtypes = [ctypes.c_int, ctypes.POINTER(_SolveOptionsC)]
_lib.Sts2Solver_Destroy.restype = None
_lib.Sts2Solver_Destroy.argtypes = [ctypes.c_int]
_lib.Sts2Solver_Reset.restype = ctypes.c_int
_lib.Sts2Solver_Reset.argtypes = [ctypes.c_int]
_lib.Sts2_SolveExpected.restype = ctypes.c_int
_lib.Sts2_SolveExpected.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_SolveOptionsC),
    ctypes.POINTER(_ExpectedResultC),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.Sts2_SolvePotionCurve.restype = ctypes.c_int
_lib.Sts2_SolvePotionCurve.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_SolveOptionsC),
    ctypes.POINTER(_CurveEntryC),
    ctypes.c_int,
]
_lib.Sts2_SolverInfo.restype = ctypes.c_int
_lib.Sts2_SolverInfo.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]


def solver_create(handle: int, options: SolveOptions | None = None, *, allow_hindsight: bool = False) -> int:
    """A solver for the fight of a combat handle; its table answers the fight's later decisions (subtree reuse).

    allow_hindsight is for the lost-fight attribution tool only (FIGHT_SOLVER.md section 4a): without it the library refuses
    Sts2_SolveHindsight.
    """
    opts = (options or SolveOptions())._c(allow_hindsight)
    out = int(_lib.Sts2Solver_Create(handle, ctypes.byref(opts)))
    if out < 0:
        raise RuntimeError(f"Sts2Solver_Create failed ({out}): {_get_last_error()}")
    return out


def solver_destroy(solver: int) -> None:
    _lib.Sts2Solver_Destroy(solver)


@contextlib.contextmanager
def open_solver(handle: int, options: SolveOptions | None = None, *, allow_hindsight: bool = False):
    """solver_create as a context manager: the solver handle is destroyed on exit."""
    solver = solver_create(handle, options, allow_hindsight=allow_hindsight)
    try:
        yield solver
    finally:
        solver_destroy(solver)


def solver_reset(solver: int) -> None:
    """Forgets the solver's tables (a new fight on the same solver); raises for an unknown solver handle."""
    if int(_lib.Sts2Solver_Reset(solver)) < 0:
        raise RuntimeError(f"Sts2Solver_Reset failed: {_get_last_error()}")


def _number(x: float):
    """A native double, None for NaN (a refused or cut result's value)."""
    x = float(x)
    return None if x != x else x


def _q_values(q, max_actions: int, stride: int) -> dict:
    values = {}
    for i in range(max_actions * stride):
        v = float(q[i])
        if v == v:  # not NaN
            values[(i // stride, i % stride - 1)] = v
    return values


def solve_expected(solver: int, handle: int, options: SolveOptions | None = None, max_actions: int = _SOLVER_MAX_ACTIONS) -> ExpectedSolve:
    """The expected value of the combat's position: exact (status 0), budget-limited (1) or refused (-3, with the reason)."""
    res = _ExpectedResultC()
    q = (ctypes.c_float * (max_actions * SOLVER_Q_STRIDE))()
    opts = options._c() if options is not None else None
    status = int(_lib.Sts2_SolveExpected(solver, handle, ctypes.byref(opts) if opts is not None else None, ctypes.byref(res), q,
                                         max_actions))
    if status == SOLVE_ERROR:
        raise RuntimeError(f"Sts2_SolveExpected failed: {_get_last_error()}")
    return ExpectedSolve(
        status=status, reason=_get_last_error() if status == SOLVE_REFUSED else None, horizon=int(res.horizon),
        value=None if status == SOLVE_REFUSED else _number(res.value), exact=bool(res.exact),
        exact_mass=None if status == SOLVE_REFUSED else _number(res.exact_mass), move_count=int(res.move_count), best_action=int(res.best_action),
        best_target=int(res.best_target), q=_q_values(q, max_actions, SOLVER_Q_STRIDE), nodes=int(res.nodes), steps=int(res.steps),
        table_hits=int(res.table_hits), leaves=int(res.leaves), horizon_leaves=int(res.horizon_leaves), seconds=float(res.seconds),
        allowed_potions=int(res.allowed_potions),
    )


def solve_potion_curve(solver: int, handle: int, options: SolveOptions | None = None, max_subsets: int = 64) -> list[dict]:
    """The potion value curve (FIGHT_SOLVER.md section 8): one entry per subset of the potions held, the empty one first.

    "subset" is a bit set over the held potions in slot order (bit j: the j-th held potion usable); "allowed_potions" is the
    same subset as a slot mask (bit i: slot i usable), the convention of SolveOptions.allowed_potions. Every entry has the
    curve's one horizon. A refused entry's numbers are None and its held map is empty.
    """
    buf = (_CurveEntryC * max_subsets)()
    opts = options._c() if options is not None else None
    n = int(_lib.Sts2_SolvePotionCurve(solver, handle, ctypes.byref(opts) if opts is not None else None, buf, max_subsets))
    if n < 0:
        raise RuntimeError(f"Sts2_SolvePotionCurve failed ({n}): {_get_last_error()}")
    out = []
    for e in buf[: min(n, max_subsets)]:
        held = {slot: float(e.held[slot]) for slot in range(8) if e.held[slot] == e.held[slot]}
        out.append({"subset": int(e.subset), "status": int(e.curve_status), "exact": bool(e.curve_exact), "horizon": int(e.curve_horizon),
                    "allowed_potions": int(e.allowed_potions), "expected_l0": _number(e.expected_l0), "win_chance": _number(e.win_chance),
                    "expected_win_hp": _number(e.expected_win_hp), "exact_mass": _number(e.curve_exact_mass), "held": held})
    return out


def solver_info() -> dict:
    """The solver's version, pruning bits and utilities (Sts2_SolverInfo)."""
    n = int(_lib.Sts2_SolverInfo(None, 0))
    buf = (ctypes.c_ubyte * max(1, n))()
    n = int(_lib.Sts2_SolverInfo(buf, n))
    return json.loads(bytes(buf[:n]).decode("utf-8"))


def solver_stats(solver: int) -> dict:
    """The last expected solve's stats on a solver (Sts2_SolverStats): nodes, engine steps, runs and moves by move kind, chance
    levels, end-turn leaves, table hits (exact, same turns left) and misses, wide-roll leaves, refusals, wall time and the
    deepening's horizons; the per-phase split and the merge audit when `solver_debug` switched them on. Bound on first use."""
    f = _lib.Sts2_SolverStats
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
    n = int(f(solver, None, 0))
    if n < 0:
        raise RuntimeError(f"Sts2_SolverStats failed ({n}): see Sts2_GetLastError")
    buf = (ctypes.c_ubyte * max(1, n))()
    n = int(f(solver, buf, n))
    return json.loads(bytes(buf[:n]).decode("utf-8"))


SOLVER_DEBUG_PHASE_TIMERS = 1
SOLVER_DEBUG_MERGE_AUDIT = 2


def solver_debug(flags: int = -1) -> int:
    """The solver's process-wide debug switches (Sts2_SolverDebug): bit 0 the phase timers, bit 1 the merge audit; a negative
    value only reads them. Returns the flags before the call. Off by default; neither changes a value. Bound on first use."""
    f = _lib.Sts2_SolverDebug
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_int]
    return int(f(int(flags)))


_hindsight_export = None


def _bind_hindsight():
    """Sts2_SolveHindsight with its signature, set on the first call to solve_hindsight (the library's flag is the gate)."""
    global _hindsight_export
    if _hindsight_export is None:
        f = _lib.Sts2_SolveHindsight
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_HindsightOptionsC), ctypes.POINTER(_HindsightResultC)]
        _hindsight_export = f
    return _hindsight_export


def solve_hindsight(solver: int, handle: int, *, node_budget: int = 0, step_budget: int = 0, time_ms: int = 0,
                    max_turns: int = 0) -> HindsightBound:
    """The hindsight bound of the combat's position with its own draws and rolls fixed, for lost-fight attribution only.

    Refused (outcome -3, with the reason) unless the solver was made with allow_hindsight=True and the handle carries the
    record's own streams (not a resampled clone, not a loaded document) with none unset (a handle from create() and reset()
    has unset streams: hindsight-unset-streams). best_utility is None unless the search was complete (outcome 0).
    """
    res = _HindsightResultC()
    opts = _HindsightOptionsC(node_budget, step_budget, time_ms, max_turns)
    outcome = int(_bind_hindsight()(solver, handle, ctypes.byref(opts), ctypes.byref(res)))
    if outcome == SOLVE_ERROR:
        raise RuntimeError(f"Sts2_SolveHindsight failed: {_get_last_error()}")
    n = min(int(res.line_length), _HINDSIGHT_LINE_CAP)
    verdict = {1: "yes", 0: "no"}.get(int(res.win_line_exists), "unknown")
    return HindsightBound(
        outcome=outcome, refusal=_get_last_error() if outcome == SOLVE_REFUSED else None, win_line_exists=verdict,
        best_utility=_number(res.best_utility) if outcome == SOLVE_EXACT else None, complete=bool(res.complete),
        line=[(int(res.line[2 * i]), int(res.line[2 * i + 1])) for i in range(n)], nodes_searched=int(res.nodes_searched),
        engine_steps=int(res.engine_steps), elapsed=float(res.elapsed),
    )
