"""The env bench's Windows counters (referee/perfcounters.py) and the bench parent's accounting (referee/envproto.py)."""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "referee"))

import envproto  # noqa: E402
import perfcounters  # noqa: E402

HOST = r"\\EXAMPLE-HOST"
HEADER = ('"(PDH-CSV 4.0)","' + '","'.join(HOST + p for p in (
    r"\Memory\Pages Input/sec", r"\Memory\Pages/sec", r"\Memory\Page Faults/sec", r"\Memory\Available MBytes",
    r"\Memory\Standby Cache Normal Priority Bytes", r"\Memory\Standby Cache Reserve Bytes", r"\Memory\Modified Page List Bytes",
    r"\Paging File(_Total)\% Usage", r"\PhysicalDisk(_Total)\Current Disk Queue Length", r"\PhysicalDisk(_Total)\% Disk Time",
    r"\Process(SlayTheSpire2)\Page Faults/sec", r"\Process(SlayTheSpire2)\Working Set", r"\Process(SlayTheSpire2)\ID Process",
    r"\Process(SlayTheSpire2#1)\Page Faults/sec", r"\Process(SlayTheSpire2#1)\Working Set", r"\Process(SlayTheSpire2#1)\ID Process")) + '"')
ROW1 = '"10/23/2026 12:00:00.100","5.5","12.0","900.0","4096.0","1073741824.0","2147483648.0","536870912.0","3.0","2.0","40.0","100.0","1048576.0","111.0","300.0","2097152.0","222.0"'
ROW2 = '"10/23/2026 12:00:02.100","7.5","14.0","1100.0","2048.0","1073741824.0","2147483648.0","536870912.0","3.0","4.0","60.0","200.0","1048576.0","111.0"' + '," ","2097152.0","222.0"'
CSV = HEADER + "\n" + ROW1 + "\n" + ROW2 + "\nExiting, please wait...\nThe command completed successfully.\n"


class TypeperfParseTest(unittest.TestCase):
    def test_canned_csv_becomes_event_fields(self):
        rows = perfcounters.parse_typeperf(CSV, pids={111})  # only pid 111 is one of the bench's games
        self.assertEqual(len(rows), 2)
        r = rows[0]
        self.assertEqual(r["pages_input_per_s"], 5.5)
        self.assertEqual(r["pages_per_s"], 12.0)
        self.assertEqual(r["page_faults_per_s"], 900.0)
        self.assertEqual(r["avail_mb"], 4096.0)
        self.assertEqual(r["standby_normal_mb"], 1024.0)  # bytes -> MB
        self.assertEqual(r["standby_reserve_mb"], 2048.0)
        self.assertEqual(r["modified_mb"], 512.0)
        self.assertEqual(r["pagefile_usage_pct"], 3.0)
        self.assertEqual((r["disk_queue"], r["disk_time_pct"]), (2.0, 40.0))
        self.assertEqual((r["game_page_faults_per_s"], r["game_working_set_mb"], r["game_procs"]), (100.0, 1.0, 1))
        self.assertEqual(rows[1]["game_procs"], 1)  # an empty cell is a missing value, not a crash

    def test_all_games_when_no_pid_filter_and_no_host_name_leaks(self):
        r = perfcounters.parse_typeperf(CSV)[0]
        self.assertEqual((r["game_page_faults_per_s"], r["game_procs"]), (400.0, 2))
        self.assertNotIn("EXAMPLE-HOST", json.dumps(r))

    def test_aggregate_is_mean_and_max(self):
        a = perfcounters.aggregate_perf(perfcounters.parse_typeperf(CSV, pids={111, 222}))
        self.assertEqual(a["perf_samples"], 2)
        self.assertEqual(a["pages_input_per_s"], 6.5)
        self.assertEqual(a["pages_input_per_s_max"], 7.5)

    def test_sysstat_carries_the_window_counters(self):
        class FakePerf:
            def drain(self):
                return {"perf_samples": 3, "pages_input_per_s": 1.0}

        acct = perfcounters.ParentAcct()
        s = envproto.sysstat(FakePerf(), acct, 4)
        self.assertEqual(s["pages_input_per_s"], 1.0)
        self.assertEqual(s["parent_qsize"], 4)
        self.assertIn("mem_load_pct", s)


class ParentAccountingTest(unittest.TestCase):
    def events(self):
        t = time.time()
        ev = [{"ev": "ready", "w": 0, "pid": 111, "boot_s": 5.0, "t": t},
              {"ev": "tick", "w": 0, "dec": 200, "t": t + 40},
              {"ev": "end", "w": 0, "seed": "s", "end": "death", "floor": 3, "asked": 30, "ms": 1, "mem_mb": 900, "dec": 30,
               "t": t + 100, "s": 5.0, "stale": None, "slowest_step": [0.1, "map"], "fight_errors": None, "probe": None}]
        for i in range(12):
            ev.append({"ev": "sys", "t": t + 10 * i, "mem_load_pct": 50, "cpu_busy_pct": 20.0, "pages_input_per_s": float(i),
                       "pages_input_per_s_max": float(i) + 1, "parent_cpu_s": 0.4, "parent_wait_s": 8.0, "parent_handle_s": 1.5,
                       "parent_net_s": 0.3, "parent_window_s": 10.0, "parent_events": 7, "parent_qsize": i % 3, "perf_samples": 5})
        return ev

    def test_summarised_window_has_the_parent_accounting(self):
        s = envproto.summarize(self.events(), 1)
        p = s["parent"]
        self.assertEqual(p["cpu_s"], 4.8)
        self.assertEqual(p["wait_s"], 96.0)
        self.assertEqual(p["handle_s"], 18.0)
        self.assertEqual(p["qsize_max"], 2)
        self.assertAlmostEqual(p["wait_share"], 0.8, places=3)
        self.assertAlmostEqual(p["cpu_share"], 0.04, places=3)

    def test_per_minute_rows_carry_decisions_counters_and_parent_sums(self):
        rows = envproto.per_minute(self.events())
        self.assertEqual(rows[0]["minute"], 0)
        self.assertEqual(rows[0]["sys_windows"], 6)
        self.assertEqual(rows[0]["parent_wait_s"], 48.0)  # summed over the six 10 s windows
        self.assertEqual(rows[0]["pages_input_per_s_max"], 6.0)
        self.assertEqual(rows[0]["dec_per_s"], round(200 / 60, 1))


if __name__ == "__main__":
    unittest.main()
