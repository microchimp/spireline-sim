"""Write the fight solver's state-key encoder (src/Sts2Emulator/Solver/StateKey.g.cs) from key_fields.json.

    python tools/solver/gen_key.py           # write the encoder
    python tools/solver/gen_key.py --check   # exit 1 when the written file is not what the json gives

docs/rebuild/FIGHT_SOLVER.md section 2. key_fields.json classes every field the full-state walk reaches from CombatState
(KeyFieldCoverageTests checks the census against reflection): ``plain`` (the value, lists in their order), ``multiset``
(a list whose order no rule reads: its elements sorted by their encoded bytes; ``ordered_when`` names the conditions that
keep it ordered), ``regions`` (the draw pile: known top, unknown middle as a multiset, known bottom), ``card_ref`` (an
int holding a card uid: encoded as where the card is, never as the number), ``stream`` (an RNG stream: in the key in
hindsight mode only) and ``excluded`` (with its ``reason``). The encoder reads fields directly, no reflection. The file
carries the json's sha256, so the coverage test catches an encoder older than its json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ROOT / "tools" / "solver" / "key_fields.json"
READERS = ROOT / "tools" / "solver" / "order_readers.json"
OUT = ROOT / "src" / "Sts2Emulator" / "Solver" / "StateKey.g.cs"

PRIMS = {"int", "bool", "long", "uint", "byte", "short", "ulong", "float", "double"}
CLASSES = {"plain", "multiset", "regions", "card_ref", "stream", "excluded"}
COLL = re.compile(r"^(List|IReadOnlyList|HashSet)<(.+)>$")


def norm(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def sha(path: Path) -> str:
    return hashlib.sha256(norm(path.read_bytes())).hexdigest()


class Gen:
    def __init__(self, spec: dict):
        self.types: dict = spec["types"]
        self.enums: set = set(spec.get("enums", []))
        self.lines: list[str] = []

    # --- element encoders --------------------------------------------------------------------------------------------

    def elem(self, t: str, v: str, w: str = "w") -> str:
        """A statement writing value ``v`` of type ``t`` to the writer named ``w``."""
        if t.endswith("?"):
            inner = t[:-1]
            if inner in PRIMS or inner in self.enums or self.is_struct(inner):
                body = self.elem(inner, f"{v}.GetValueOrDefault()", w)
                return f"if ({v}.HasValue) {{ {w}.Has(true); {body} }} else {w}.Has(false);"
            return self.elem(inner, v, w)
        if t in PRIMS:
            return f"{w}.P({v});"
        if t in self.enums:
            return f"{w}.P((long){v});"
        if t == "string":
            return f"{w}.S({v});"
        if t.endswith("[]"):
            return self.seq(t[:-2], v, False, w)
        m = COLL.match(t)
        if m:
            return self.seq(m.group(2), v, m.group(1) == "HashSet", w)
        if t == "CardInstance":
            return f"{w}.Card({v});"
        if t == "CombatState":
            return f"if ({v} is null) {w}.Has(false); else {{ {w}.Has(true); {w}.Combat({v}); }}"
        if t in self.types:
            if self.is_struct(t):
                return f"Enc_{t}({w}, {v});"
            return f"if ({v} is null) {w}.Has(false); else {{ {w}.Has(true); Enc_{t}({w}, {v}); }}"
        raise SystemExit(f"gen_key: no encoder for type {t} ({v}); add it to key_fields.json")

    def is_struct(self, t: str) -> bool:
        return t in self.types and self.types[t].get("kind") == "struct"

    def seq(self, et: str, v: str, multiset: bool, w: str = "w") -> str:
        # nested lambdas get their own names (w1/e1, w2/e2, ...)
        self.depth = getattr(self, "depth", 0) + 1
        d = self.depth
        e = self.elem(et, f"e{d}", f"w{d}")
        self.depth -= 1
        call = "Bag" if multiset else "Seq"
        return f"{w}.{call}({v}, static (w{d}, e{d}) => {{ {e} }});"

    # --- fields ------------------------------------------------------------------------------------------------------

    def field(self, owner: str, name: str, f: dict) -> list[str]:
        cls = f["class"]
        if cls not in CLASSES:
            raise SystemExit(f"gen_key: {owner}.{name}: unknown class {cls}")
        if cls == "excluded":
            if not f.get("reason"):
                raise SystemExit(f"gen_key: {owner}.{name}: an exclusion needs a reason")
            return [f"        // {name}: excluded ({f['reason'][:90]})"]
        if name.endswith("Uid") and cls != "card_ref" and f["type"] == "int":
            raise SystemExit(f"gen_key: {owner}.{name}: an int field named *Uid must be classed card_ref or excluded")
        v = "x." + f.get("via", name)
        t = f["type"]
        if cls == "plain":
            return [f"        {self.elem(t, v)}"]
        if cls == "card_ref":
            return [f"        w.CardRef({v});"]
        if cls == "stream":
            return [f"        if (w.Hindsight) w.Stream({v});"]
        if cls == "regions":
            return [f"        w.Regions({v}, x.KnownTopCount, x.KnownBottomCount);"]
        # multiset, ordered under the named conditions
        m = COLL.match(t)
        et = m.group(2) if m else t[:-2]
        conds = f.get("ordered_when", [])
        flags = " || ".join(f"w.{c}" for c in conds) or "false"
        return [f"        if ({flags}) {self.seq(et, v, False)} else {self.seq(et, v, True)}"]

    def run(self, digest: str, readers: dict) -> str:
        L = self.lines
        L += [
            "// <auto-generated> by tools/solver/gen_key.py from tools/solver/key_fields.json; do not edit. </auto-generated>",
            "// The fight solver's state key (docs/rebuild/FIGHT_SOLVER.md section 2): one Enc_ method per classed type, fields",
            "// in the json's order. The hand-written half (Solver/StateKey.cs) holds the writer, the frames and the forms.",
            "#nullable enable",
            "using Sts2Emulator.Core;",
            "using Sts2Emulator.Core.Rng;",
            "using Sts2Emulator.Core2;",
            "",
            "namespace Sts2Emulator.Solver;",
            "",
            "internal static partial class StateKeyCodec",
            "{",
            f'    internal const string FieldsSha256 = "{digest}";',
            "",
        ]
        for cat in ("hand_position", "potion_slot", "enemy_order"):
            r = readers["readers"][cat]
            names = ", ".join(f'"{c["name"]}"' for c in r.get("cards", []))
            L.append(f"    internal static readonly string[] {''.join(p.title() for p in cat.split('_'))}Cards = [{names}];")
            L.append(f"    internal const bool {''.join(p.title() for p in cat.split('_'))}Always = {str(bool(r.get('always'))).lower()};")
        L.append("")
        for tname, tspec in self.types.items():
            ns = tspec.get("cs", tname)
            sig = f"in {ns} x" if tspec.get("kind") == "struct" else f"{ns} x"
            L.append(f"    internal static void Enc_{tname}(KeyWriter w, {sig})")
            L.append("    {")
            for fname, f in tspec["fields"].items():
                L += self.field(tname, fname, f)
            L.append("    }")
            L.append("")
        L.append("}")
        return "\n".join(L) + "\n"


def build() -> str:
    spec = json.loads(FIELDS.read_text(encoding="utf-8"))
    readers = json.loads(READERS.read_text(encoding="utf-8"))
    digest = hashlib.sha256(norm(FIELDS.read_bytes()) + norm(READERS.read_bytes())).hexdigest()
    return Gen(spec).run(digest, readers)


def main(argv: list[str]) -> int:
    text = build()
    if "--check" in argv:
        have = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if have.replace("\r\n", "\n") != text:
            print("gen_key: StateKey.g.cs is stale; run python tools/solver/gen_key.py", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"gen_key: wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
