"""A stand-in for one referee game server (referee/driver/Server.cs), for the serve.Pool tests.

    python fake_fight_server.py <job.json>

Speaks the same inbox protocol: writes hb/<pid>.ready, keeps hb/<pid> fresh, waits while hold.<pid>
exists, claims the oldest *.json by renaming it to *.json.taken.<pid>, appends one result line per
fight to the job's "out" file, deletes the claim, and exits 0 once STOP is in the inbox and no
*.json is left. A spec's "fake" key says how the fight goes:
  {"sleep": s}         play for s seconds (default 0.02)
  {"crash": "always"}  exit 3 holding the claim; {"crash": "once"}: only the first time
  {"hang": "once"}     stop heartbeating and wait to be killed, the first time
  {"fatal": "once"}    write a server-error line ({"_fatal": id}) and exit 1, the first time
  {"bad": true}        answer like a spec file that could not be read (an error line with no seed)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def first_time(inbox: Path, rid: str, what: str) -> bool:
    mark = inbox.parent / f"fake.{what}.{rid}"
    if mark.exists():
        return False
    mark.write_text("", encoding="utf-8")
    return True


def pending(inbox: Path) -> list:
    """Pending specs, oldest first; a file another server renames away meanwhile is skipped."""
    found = []
    for f in inbox.iterdir():
        if f.name.endswith(".json"):
            try:
                found.append((f.stat().st_mtime, f.name, f))
            except FileNotFoundError:
                continue
    return [f for _, _, f in sorted(found)]


def read(path: Path) -> str:
    """The claimed spec. A virus scanner may hold a file just written under %TEMP% for a moment
    (PermissionError, about 1 in 2000 reads here; never seen with the game's driver), so that is
    retried; a missing file is not (that is the double-claim signature the tests look for)."""
    for _ in range(50):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            time.sleep(0.01)
    return path.read_text(encoding="utf-8")


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    inbox, out = Path(job["inbox"]), Path(job["out"])
    pid = os.getpid()
    hb = inbox / "hb" / str(pid)
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text("idle", encoding="utf-8")
    (inbox / "hb" / f"{pid}.ready").write_text(str(int(time.time() * 1000)), encoding="utf-8")
    hold, stop = inbox / f"hold.{pid}", inbox / "STOP"
    while True:
        hb.write_text("idle", encoding="utf-8")
        claim = None
        if not hold.exists():
            for f in pending(inbox):
                taken = f.with_name(f.name + f".taken.{pid}")
                try:
                    os.rename(f, taken)
                except OSError:
                    continue
                claim = (taken, f.name[: -len(".json")])
                break
        if claim is None:
            if stop.exists() and not pending(inbox):
                return 0
            time.sleep(0.01)
            continue
        taken, rid = claim
        try:
            spec = json.loads(read(taken))
        except OSError as e:  # as Server.cs: a claim it cannot read is answered as a bad spec file
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"error": f"bad spec file: {e}", "_id": rid, "_pid": pid, "_t": int(time.time() * 1000)}) + "\n")
            continue
        fake = spec.get("fake", {})
        if fake.get("crash") == "always" or (fake.get("crash") == "once" and first_time(inbox, rid, "crash")):
            return 3
        if fake.get("hang") == "once" and first_time(inbox, rid, "hang"):
            time.sleep(3600)
        if fake.get("fatal") == "once" and first_time(inbox, rid, "fatal"):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"_fatal": rid, "error": "run-start exception"}) + "\n")
            return 1
        if fake.get("bad"):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"error": "bad spec file: gone", "_id": rid, "_pid": pid, "_t": int(time.time() * 1000)}) + "\n")
            taken.unlink()
            continue
        time.sleep(fake.get("sleep", 0.02))
        result = {"seed": spec.get("seed"), "steps": [{"i": i} for i in range(spec.get("n_steps", 3))],
                  "timing_ms": {"play": 5, "reset": 1}, "cost": {"cpu_ms": 1}, "_id": rid, "_pid": pid,
                  "_t": int(time.time() * 1000)}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        taken.unlink()


if __name__ == "__main__":
    sys.exit(main())
