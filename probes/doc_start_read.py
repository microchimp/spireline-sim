"""Read the per-door diagnosis files: the first difference between the two ways, by item.

    python -I doc_start_read.py DIAG_DIR
"""
import json
import sys
from pathlib import Path


def main():
    for p in sorted(Path(sys.argv[1]).glob("*.json")):
        r = json.loads(p.read_text())
        print("==", p.name, "hp", r["hp"], "/", r["max_hp"], "cfg", r["cfg"])
        print(" encounter ids", r["encounter_id"], "doc", r["encounter_id_doc"])
        print(" obs equal", r["obs_equal"], r["obs_diff"][:10])
        print(" tokens noprobe", {k: v for k, v in r["tokens_noprobe"].items() if k != "feat_diffs"}, r["tokens_noprobe"].get("feat_diffs", [])[:5])
        print(" tokens probe  ", {k: v for k, v in r["tokens_probe"].items() if k != "feat_diffs"}, r["tokens_probe"].get("feat_diffs", [])[:5])
        print(" legal equal", r["legal_run"] == r["legal_doc"], "moves equal", r["moves_run"] == r["moves_doc"], len(r["moves_run"]))
        print(" pos", r["pos_run"], "==" if r["pos_run"] == r["pos_doc"] else r["pos_doc"])
        print(" kwargs equal", r["kwargs_run"] == r["kwargs_doc"], r["kwargs_run"] if r["kwargs_run"] != r["kwargs_doc"] else "")
        print(" kwargs", r["kwargs_run"])
        for s in r["searches"]:
            print("  search seed", s["seed"], "run", s["run"]["sims"], s["run"]["exhausted"], s["run"]["assumed"], s["run"]["best"],
                  "| doc", s["doc"]["sims"], s["doc"]["exhausted"], s["doc"]["assumed"], s["doc"]["best"],
                  "| steps", s["doc_steps"]["sims"], s["doc_steps"]["best"], "identical live/steps:", s["doc"]["children"] == s["doc_steps"]["children"])
        for g in r["games"]:
            print(" game", g["j"], "run:", {k: v for k, v in g["run"].items() if k != "searches"}, "doc:", {k: v for k, v in g["doc"].items() if k != "searches"})
            a, b = g["run"]["searches"], g["doc"]["searches"]
            print("   searches run/doc", len(a), len(b), "first run/doc", {k: a[0][k] for k in ("kwargs", "tok_sha", "legal", "sims", "exhausted", "assumed", "elapsed")}, {k: b[0][k] for k in ("kwargs", "tok_sha", "legal", "sims", "exhausted", "assumed", "elapsed")})
        et = r["endturn"]
        for name in ("oracle_run", "oracle_doc"):
            print(" ", name, json.dumps(et[name])[:700])


if __name__ == "__main__":
    main()
