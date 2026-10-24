"""Damage folds against the old card pre-calculation (docs/rebuild/MIGRATION_F08.md, "Damage terms applied twice").

    python -m tools.kernel.fold_precalc

For every model that registers a damage fold (ModifyDamageAdditive, ModifyDamageMultiplicative, ModifyDamageCap, the
ModifyHpLost passes) under src/Sts2Emulator/Core2/Models, lists the old engine's calls of that model's extracted body
(``<Model>_<Hook>(``, round R5's names) with the method each sits in and whether an F08 switch test comes before it in
that method. A call in a method with no F08 test is a pre-calculation candidate: its callers decide whether the amount
it builds reaches Core2's damage command (applied twice) or only the old branch. Bodies the old engine names otherwise
(Conqueror's ConqueredDamage, Mystic Lighter's EnchantedCardDamageBonus, the enchantment's EnchantDamageAdditive and
EnchantDamageMultiplicative) are read by hand, as the doc's table says. Identifiers only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ["ModifyDamageAdditive", "ModifyDamageMultiplicative", "ModifyDamageCap", "ModifyHpLostBeforeOsty",
         "ModifyHpLostBeforeOstyLate", "ModifyHpLostAfterOsty", "ModifyHpLostAfterOstyLate"]
METHOD = re.compile(r"\s*(?:private|internal|public|protected)[\w\s<>\[\],?]*\s(\w+)\s*\(")


def pairs(models: Path) -> list[tuple[str, str]]:
    out = []
    for f in sorted(models.rglob("*.cs")):
        text = f.read_text(encoding="utf-8")
        m = re.search(r't\.Model\(ListenerKind\.\w+, "(\w+)"\)', text)
        for hook in HOOKS:
            if re.search(r"HookId\.%s\b" % hook, text):
                out.append((m.group(1) if m else f.stem, hook))
    return out


def old_calls(core: Path, model: str, hook: str) -> list[str]:
    base = model.removesuffix("Power")
    call = re.compile(r"\b%s_%s\w*\(" % (re.escape(base), hook.removesuffix("Late")))
    found = []
    for p in sorted(core.rglob("*.cs")):
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if not call.search(line) or s.startswith("//") or re.search(r"static \w+ %s_" % re.escape(base), line):
                continue
            start, name = 0, "?"
            for j in range(i, -1, -1):
                if mm := METHOD.match(lines[j]):
                    start, name = j, mm.group(1)
                    break
            gated = any("Switch.F08" in lines[k] for k in range(start, i))
            found.append(f"{p.name}:{name} (F08 test before: {'yes' if gated else 'no'})")
    return found


def main() -> int:
    src = ROOT / "src" / "Sts2Emulator"
    for model, hook in pairs(src / "Core2" / "Models"):
        calls = old_calls(src / "Core", model, hook)
        print(f"{model} | {hook} | " + ("; ".join(calls) if calls else "no named old call"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
