"""Windows performance counters for the env bench (referee/envproto.py), and the bench parent's own accounting.

One `typeperf` child of the bench parent samples memory, paging, disk and the games' own page faults and working
sets every 2 s. Its samples are averaged into each `sys` event (field = mean, field_max = max). The Process
counters are summed over the bench's own game pids only (the lab may run games too). Results hold no host names:
typeperf's header carries the host in every counter path and the parser drops it.
"""

from __future__ import annotations

import csv
import re
import subprocess
import threading
import time

_MB = 1.0 / 1048576
PERF_SYSTEM = {  # counter path without the host -> (event field, scale)
    r"\Memory\Pages Input/sec": ("pages_input_per_s", 1.0),
    r"\Memory\Pages/sec": ("pages_per_s", 1.0),
    r"\Memory\Page Faults/sec": ("page_faults_per_s", 1.0),
    r"\Memory\Available MBytes": ("avail_mb", 1.0),
    r"\Memory\Standby Cache Normal Priority Bytes": ("standby_normal_mb", _MB),
    r"\Memory\Standby Cache Reserve Bytes": ("standby_reserve_mb", _MB),
    r"\Memory\Modified Page List Bytes": ("modified_mb", _MB),
    r"\Paging File(_Total)\% Usage": ("pagefile_usage_pct", 1.0),
    r"\PhysicalDisk(_Total)\Current Disk Queue Length": ("disk_queue", 1.0),
    r"\PhysicalDisk(_Total)\% Disk Time": ("disk_time_pct", 1.0),
}
PERF_PROCESS = (r"\Process(SlayTheSpire2*)\Page Faults/sec", r"\Process(SlayTheSpire2*)\Working Set",
                r"\Process(SlayTheSpire2*)\ID Process")
_HOST_PREFIX = re.compile(r"^\\\\[^\\]+")  # the \\HOST in a typeperf header path
_PROC_PATH = re.compile(r"^\\Process\(([^)]*)\)\\(.+)$")


def _num(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


class TypeperfParser:
    """typeperf's CSV: a header of counter paths, then one row per sample. row() -> the event fields, or None."""

    def __init__(self, header: str, pids: set | None = None):
        self.pids = pids  # the bench's game pids: only those games' Process instances are summed
        self.cols: list = []
        for c in next(csv.reader([header]))[1:]:
            path = _HOST_PREFIX.sub("", c.strip())
            m = _PROC_PATH.match(path)
            if m:
                self.cols.append(("proc", m.group(1), m.group(2), 1.0))
            elif path in PERF_SYSTEM:
                self.cols.append(("sys", PERF_SYSTEM[path][0], "", PERF_SYSTEM[path][1]))
            else:
                self.cols.append(("skip", path, "", 1.0))

    def row(self, line: str) -> dict | None:
        f = next(csv.reader([line]), [])
        if len(f) != len(self.cols) + 1:
            return None  # typeperf's closing lines ("Exiting, please wait...") or a torn line
        out: dict = {}
        inst: dict[str, dict] = {}
        for (kind, a, b, _scale), x in zip(self.cols, f[1:]):
            v = _num(x)
            if kind == "sys":
                out[a] = None if v is None else round(v * _scale, 3)
            elif kind == "proc":
                inst.setdefault(a, {})[b] = v
        faults = ws = 0.0
        n = 0
        for d in inst.values():
            pid = d.get("ID Process")
            if self.pids is not None and (pid is None or int(pid) not in self.pids):
                continue
            faults += d.get("Page Faults/sec") or 0.0
            ws += d.get("Working Set") or 0.0
            n += 1
        if inst:
            out.update(game_page_faults_per_s=round(faults, 1), game_working_set_mb=round(ws * _MB, 1), game_procs=n)
        return out


def parse_typeperf(text: str, pids: set | None = None) -> list[dict]:
    """A whole typeperf CSV (header line, then samples) -> one dict of event fields per sample."""
    parser = None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if parser is None:
            if line.startswith('"(PDH-CSV'):
                parser = TypeperfParser(line, pids)
            continue
        r = parser.row(line)
        if r is not None:
            rows.append(r)
    return rows


def aggregate_perf(rows: list[dict]) -> dict:
    """The samples of one sys window: each field's mean, and its max as <field>_max."""
    out: dict = {"perf_samples": len(rows)}
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != "t"})
    for k in keys:
        vs = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = round(sum(vs) / len(vs), 2)
        out[k + "_max"] = round(max(vs), 2)
    return out


class PerfSampler:
    """One typeperf child of the bench parent (every `interval` s); its samples are drained into each sys event.

    typeperf expands the Process wildcard once, at start, so a game that boots later is not in it: the child is
    restarted (at most every 30 s) when a pid it has not seen is ready.
    """

    def __init__(self, pids: set, interval: int = 2):
        self.pids, self.interval = pids, interval
        self.proc = None
        self.known: set = set()
        self.t_start = 0.0
        self.rows: list = []
        self.lock = threading.Lock()
        self.starts = 0

    def start(self) -> None:
        args = ["typeperf", "-si", str(self.interval), *PERF_SYSTEM, *PERF_PROCESS]
        try:
            self.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        except OSError:
            self.proc = None
            return
        self.known, self.t_start = set(self.pids), time.time()
        self.starts += 1
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc) -> None:
        parser = None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if parser is None:
                if line.startswith('"(PDH-CSV'):
                    parser = TypeperfParser(line, self.pids)
                continue
            r = parser.row(line) if line else None
            if r is not None:
                r["t"] = time.time()
                with self.lock:
                    self.rows.append(r)

    def poll(self) -> None:
        if self.pids - self.known and time.time() - self.t_start > 30:
            self.stop()
            self.start()

    def drain(self) -> dict:
        with self.lock:
            rows, self.rows = self.rows, []
        return aggregate_perf(rows)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc = None


class ParentAcct:
    """The bench parent's own accounting for one sys window: CPU seconds, blocked on the result queue, handling."""

    def __init__(self):
        self.cpu0, self.t0 = time.process_time(), time.perf_counter()
        self.wait = self.handle = self.net = 0.0
        self.events = 0

    def snap(self, qsize: int | None = None) -> dict:
        cpu, now = time.process_time(), time.perf_counter()
        out = {"parent_cpu_s": round(cpu - self.cpu0, 3), "parent_wait_s": round(self.wait, 3),
               "parent_handle_s": round(self.handle, 3), "parent_net_s": round(self.net, 3),
               "parent_window_s": round(now - self.t0, 3), "parent_events": self.events, "parent_qsize": qsize}
        self.cpu0, self.t0, self.wait, self.handle, self.net, self.events = cpu, now, 0.0, 0.0, 0.0, 0
        return out
