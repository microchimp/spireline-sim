"""Replay a run to its boss doors and save the exported document at each (explore step; the diagnosis proper is doc_start_diag.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402


def main():
    seed, ix, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    L.lower()
    from sts2_gym import native
    player = L.build_player()

    def on_door(s, info):
        fight = s.combat(fork_combat=True)
        root = fight.combat_fork(None)
        try:
            doc = native.get_state_json(root.handle)
        finally:
            root.close()
        (out / f"{seed}_f{info['floor']}.json").write_text(json.dumps({"seed": seed, "ix": ix, "floor": int(info["floor"]), "info": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in info.items()}, "doc": doc}))
        print("saved", seed, info["floor"], flush=True)
        return True

    L.walk_doors(seed, ix, player, on_door, acts=(0,))


if __name__ == "__main__":
    main()
