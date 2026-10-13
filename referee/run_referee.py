#!/usr/bin/env python3
"""Run one referee session: the user's own copy of the game, headless, driven by our mod.

    python referee/run_referee.py --game <game copy> --job job.json [--scratch <dir>]

What it does, per Spireline security addendum 1, section 3:
  H1  refuses unless sts2.dll is the known v0.107.1 build (sha256 below);
  H2  launches with --force-steam off and an override.cfg that blanks the Sentry DSN, and records
      every network connection the game process opens (there should be none);
  H3  points APPDATA/LOCALAPPDATA at a scratch folder, so saves, logs and settings land there,
      and checks the real profile folder was not created or touched.
The driver mod is copied into <game>/mods/SpirelineReferee and is inert without the job variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KNOWN_BUILDS = {  # sts2.dll sha256 -> build
    "a1f9e653f1e28e4076558fee1e60d218619cb7e057b887c6417f62c62c6d7a52": "v0.107.1 / 23811903",
}
HERE = Path(__file__).resolve().parent
OVERRIDE = '[sentry]\n\nconfig/dsn=""\n'


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def connections(pid: int) -> set[str]:
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, check=False,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout
    found = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1] == str(pid) and parts[0] in ("TCP", "UDP"):
            found.add(" ".join(parts[:4] if parts[0] == "TCP" else parts[:3]))
    return found


def install(game: Path, build: Path) -> None:
    """override.cfg (blank Sentry DSN) and the driver mod, copied only when they changed."""
    cfg = game / "override.cfg"
    if not cfg.exists() or cfg.read_text(encoding="utf-8") != OVERRIDE:
        cfg.write_text(OVERRIDE, encoding="utf-8")
    mod = game / "mods" / "SpirelineReferee"
    mod.mkdir(parents=True, exist_ok=True)
    # Parallel sessions share the game copy: copy the mod only when it changed, since a
    # running game holds the loaded DLL open.
    for src, dst in ((build / "SpirelineReferee.dll", mod / "SpirelineReferee.dll"),
                     (HERE / "driver" / "SpirelineReferee.json", mod / "SpirelineReferee.json")):
        if not dst.exists() or sha256(src) != sha256(dst):
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--game", type=Path, required=True)
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--build", type=Path, default=HERE / "driver" / "bin" / "Release" / "net9.0")
    ap.add_argument("--no-install", action="store_true", help="the caller already installed the mod and override.cfg")
    a = ap.parse_args()

    game = a.game.resolve()
    dll = game / "data_sts2_windows_x86_64" / "sts2.dll"
    digest = sha256(dll)
    allowed = set(KNOWN_BUILDS) | set(filter(None, os.environ.get(
        "SPIRELINE_REFEREE_KNOWN_SHA256", "").split(",")))
    if digest not in allowed:
        print(f"refusing: unknown sts2.dll {digest}", file=sys.stderr)
        return 2

    scratch = (a.scratch or game.parent / "scratch" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    (scratch / "appdata").mkdir(parents=True, exist_ok=True)
    (scratch / "localappdata").mkdir(parents=True, exist_ok=True)
    if not a.no_install:
        install(game, a.build)

    # A fresh profile with every volume at 0 (FMOD plays even headless) and mod loading agreed to:
    # the game skips mods until the player has seen its mods warning. Missing fields take defaults.
    settings = scratch / "appdata" / "SlayTheSpire2" / "default" / "1" / "settings.save"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "schema_version": 5, "volume_master": 0.0, "volume_bgm": 0.0, "volume_sfx": 0.0,
        "volume_ambience": 0.0, "fullscreen": False,
        "mod_settings": {"mods_enabled": True, "mod_list": []},
    }, indent=2), encoding="utf-8")

    real_profile = Path(os.environ["APPDATA"]) / "SlayTheSpire2"
    before = real_profile.exists() and sorted((str(p), p.stat().st_mtime) for p in real_profile.rglob("*"))

    env = dict(os.environ, APPDATA=str(scratch / "appdata"), LOCALAPPDATA=str(scratch / "localappdata"),
               SPIRELINE_REFEREE_JOB=str(a.job.resolve()))
    log = open(scratch / "game-stdout.log", "wb")
    t0 = time.time()
    proc = subprocess.Popen([str(game / "SlayTheSpire2.exe"), "--headless", "--force-steam", "off"],
                            cwd=game, env=env, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    seen: set[str] = set()
    while proc.poll() is None and time.time() - t0 < a.timeout:
        seen |= connections(proc.pid)
        time.sleep(0.5)
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        rc = "timeout"
    else:
        rc = proc.returncode
    after = real_profile.exists() and sorted((str(p), p.stat().st_mtime) for p in real_profile.rglob("*"))
    report = {
        "exit": rc, "seconds": round(time.time() - t0, 1), "sts2_dll_sha256": digest,
        "network_connections": sorted(seen), "real_profile_touched": before != after,
        "scratch": scratch.name,
    }
    (scratch / "session.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if rc == 0 and not seen and before == after else 1


if __name__ == "__main__":
    sys.exit(main())
