#!/usr/bin/env python3
"""Event option keys of the human run record to the copy's option index (D55, 2026-11-05).

    python rl/event_keys.py resolve ABYSSAL_BATHS ABYSSAL_BATHS.pages.INITIAL.options.IMMERSE.title
    python rl/event_keys.py coverage --counts COUNTS.json      # the counts file of scripts/build_event_option_titles.py scan

The human record (Spire Codex public run export for v0.107.1) names the option a player took by a text key,
`<EVENT>.pages.<PAGE>.options.<OPTION>.title`; the copy takes an option by its index on the page (the action number). The table that joins
them is `data/event_option_titles.json`, built by `scripts/build_event_option_titles.py` from the live captures, the engine's event code
and the record's own keys and outcomes (no decompiled text). `resolve` answers with the index, or None when the table cannot say:

    resolve(event_id, record_title, state=None) -> index | None

  event_id      the game's id (`ABYSSAL_BATHS`), the class name (`AbyssalBaths`) or the copy's event id (20)
  record_title  the record's full key; `PAGE.OPTION`; the bare option name (`IMMERSE`); a relic-titled choice (`BING_BONG.title`); or
                the title the game shows (`Give 100 Gold`: a title with a number or a dish matches by its pattern)
  state         optional dict: `page` (the record's page name when `record_title` has none) and `offered` (the options on the page in
                display order, as option names: needed for pages whose options are rolled per visit -- the dolls, Tinker Time's
                types and riders, Colorful Philosophers' pools, the trades -- and ignored for the rest)

Fixed options resolve on their own. Anything the table does not know (an event of another mod, an option no capture or engine option
carries, a rolled page without `offered`) is None, never a guess.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "event_option_titles.json"
KEY = re.compile(r"^([A-Z0-9_]+)\.pages\.([A-Z0-9_]+)\.options\.([A-Z0-9_]+)\.title$")
RELIC_TITLE = re.compile(r"^([A-Z0-9_]+)\.title$")
PAGE_OPTION = re.compile(r"^([A-Z0-9_]+)\.([A-Z0-9_]+)$")
BARE = re.compile(r"^[A-Z0-9_]+$")
MIN_CELL = 20  # D13: a cell over fewer visits is not shown
# How far a row's index is from a capture, strongest first; a visit or an event is as weak as its weakest key.
CLASSES = ("capture", "engine", "inferred", "dynamic", "unresolved")


def _slug(name: str) -> str:
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[A-Z])", "_", str(name).strip())
    return re.sub(r"[^A-Z0-9_]", "", re.sub(r"\s+", "_", text.upper()))


def row_class(row: Mapping) -> str:
    """The strength of one table row: capture (the name's words are in a captured title), engine (the engine's option order),
    inferred (a capture's lock text or the record's outcomes), dynamic (rolled per visit) or unresolved."""
    if row.get("dynamic"):
        return "dynamic"
    if row["mapping"] == "unresolved" or row.get("index") is None:
        return "unresolved"
    if row["mapping"] == "exact":
        return "capture"
    return "engine" if row["source"] == "engine" else "inferred"


class EventTable:
    def __init__(self, doc: Mapping) -> None:
        self.doc = doc
        self.by_name: dict[str, dict] = dict(doc["events"])
        self.by_id: dict[int, str] = {int(e["engine_id"]): name for name, e in self.by_name.items()}
        self._compiled: dict[str, list[tuple[re.Pattern, list[dict]]]] = {}

    @classmethod
    def load(cls, path: Path | str = DATA) -> EventTable:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ----------------------------------------------------------------------------------------------------------------- lookup
    def name(self, event_id) -> str | None:
        """The table's name for an event id given as the game's id, a class name or the copy's id."""
        if isinstance(event_id, int) or (isinstance(event_id, str) and event_id.lstrip("-").isdigit()):
            return self.by_id.get(int(event_id))
        text = str(event_id)
        if text.startswith("EVENT."):
            text = text[6:]
        if text in self.by_name:
            return text
        text = _slug(text)
        return text if text in self.by_name else None

    def _pages(self, name: str) -> list[tuple[re.Pattern, list[dict]]]:
        if name not in self._compiled:
            out = []
            for page in self.by_name[name]["pages"]:
                pat = re.compile(page["page"] if page.get("page_pattern") else re.escape(page["page"]))
                rows = []
                for r in page["options"]:
                    rr = dict(r)
                    rr["_key_re"] = re.compile(r["key_pattern"]) if r.get("key_pattern") else None
                    rr["_title_re"] = re.compile(r["title_pattern"], re.IGNORECASE) if r.get("title_pattern") else None
                    rr["_page"] = page["page"]
                    rows.append(rr)
                out.append((pat, rows))
            self._compiled[name] = out
        return self._compiled[name]

    @staticmethod
    def parse(record_title: str) -> tuple[str | None, str | None, str | None, str | None]:
        """(event named in the title, page, option name, display title): the shapes of `record_title`."""
        text = str(record_title).strip()
        m = KEY.match(text)
        if m:
            return m.group(1), m.group(2), m.group(3), None
        m = RELIC_TITLE.match(text)
        if m:
            return None, None, m.group(1), None
        m = PAGE_OPTION.match(text)
        if m:
            return None, m.group(1), m.group(2), None
        if BARE.match(text):
            return None, None, text, None
        return None, None, None, text

    def rows_for(self, event_id, record_title: str, state: Mapping | None = None) -> list[dict]:
        """The table rows a record title can mean (none: unknown; several: the same name on several pages)."""
        state = state or {}
        ev_in_key, page, opt, display = self.parse(record_title)
        name = self.name(event_id) if event_id is not None else None
        if name is None and ev_in_key:
            name = self.name(ev_in_key)
        if name is None:
            return []
        if ev_in_key and self.name(ev_in_key) != name:
            return []  # a key of another event than the one asked
        page = page or state.get("page")
        pages = self._pages(name)
        hits: list[dict] = []
        for pat, rows in pages:
            for r in rows:
                if display is not None:
                    title = r.get("title")
                    ok = (title is not None and title.casefold() == display.casefold()) or (r["_title_re"] is not None and r["_title_re"].fullmatch(display))
                else:
                    ok = r["key"] == opt or (r["_key_re"] is not None and r["_key_re"].fullmatch(opt or ""))
                if ok:
                    hits.append(dict(r, _page_re=pat))
        if display is None and any(r["key"] == opt for r in hits):
            hits = [r for r in hits if r["key"] == opt]  # a name written out beats a pattern that also fits it
        elif display is not None and any((r.get("title") or "").casefold() == display.casefold() for r in hits):
            hits = [r for r in hits if (r.get("title") or "").casefold() == display.casefold()]  # likewise for a title
        if page is not None and len(hits) > 1:
            on_page = [r for r in hits if r["_page_re"].fullmatch(page)]
            hits = on_page or hits
        return hits

    def resolve(self, event_id, record_title: str, state: Mapping | None = None) -> int | None:
        """The option index the record title means on the copy, or None (see the module docstring)."""
        state = state or {}
        hits = self.rows_for(event_id, record_title, state)
        if not hits:
            return None
        fixed = {r["index"] for r in hits if r.get("index") is not None and not r.get("dynamic")}
        dyn = [r for r in hits if r.get("dynamic")]
        if dyn and not fixed:
            offered = state.get("offered")
            if not offered:
                return None
            _, _, opt, _ = self.parse(record_title)
            offered = [str(x) for x in offered]
            return offered.index(opt) if opt in offered else None
        return next(iter(fixed)) if len(fixed) == 1 and not dyn else None

    def followup(self, event_id, last_key: str) -> int | None:
        """The index of an option the record does not list (the game takes it as the continuation of the choice before it), or None.
        `last_key` is the option name or full key of the choice it follows: Round Tea Party's warning page follows PICK_FIGHT."""
        name = self.name(event_id)
        if name is None:
            return None
        _, _, opt, _ = self.parse(last_key)
        for page in self.by_name[name]["pages"]:
            for r in page["options"]:
                if r.get("recorded") is False and r.get("after_key") == opt and r.get("index") is not None:
                    return r["index"]
        return None

    def describe(self, event_id, record_title: str, state: Mapping | None = None) -> dict | None:
        """The table row for a title (the strongest class when the name sits on several pages), for reports."""
        hits = self.rows_for(event_id, record_title, state)
        return min(hits, key=lambda r: CLASSES.index(row_class(r))) if hits else None


@functools.lru_cache(maxsize=1)
def table(path: str = str(DATA)) -> EventTable:
    return EventTable.load(path)


def resolve(event_id, record_title: str, state: Mapping | None = None) -> int | None:
    """Module-level `EventTable.resolve` over the shipped table."""
    return table().resolve(event_id, record_title, state)


# --------------------------------------------------------------------------------------------------------------------- coverage
def _key_row(tab: EventTable, ev: str, k: str) -> dict | None:
    """The row of a scan key (`PAGE.OPTION`, or `~NAME` for a relic-titled choice)."""
    title = k[1:] + ".title" if k.startswith("~") else k
    return tab.describe(ev, title)


def coverage(tab: EventTable, seqs: Iterable[tuple[str, list[str], int]], visits: Mapping[str, int] | None = None) -> dict:
    """What share of the record's event decisions the table covers.

    `seqs` are (event, option keys of one visit in order, number of such visits). A decision is one chosen key; a visit is as weak as its
    weakest key. An event's class is its weakest key among the keys chosen at least MIN_CELL times. Returns plain counts (suppress
    cells under MIN_CELL before quoting a per-event row)."""
    per_event: dict[str, Counter] = defaultdict(Counter)
    key_n: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for ev, keys, n in seqs:
        if tab.name(ev) is None:
            per_event[ev]["visits_other"] += n
            totals["visits_other"] += n
            continue
        if not keys:
            per_event[ev]["visits_no_choice"] += n
            totals["visits_no_choice"] += n
            continue
        weakest = 0
        for k in keys:
            row = _key_row(tab, ev, k)
            cls = row_class(row) if row else "unresolved"
            per_event[ev]["decisions_" + cls] += n
            totals["decisions_" + cls] += n
            key_n[ev][(k, cls)] += n
            weakest = max(weakest, CLASSES.index(cls))
        per_event[ev]["visits_" + CLASSES[weakest]] += n
        totals["visits_" + CLASSES[weakest]] += n
    event_class = {}
    for ev, kc in key_n.items():
        big = [CLASSES.index(c) for (_, c), n in kc.items() if n >= MIN_CELL] or [CLASSES.index(c) for (_, c) in kc]
        event_class[ev] = CLASSES[max(big)]
    return {"per_event": {e: dict(c) for e, c in per_event.items()}, "totals": dict(totals), "event_class": event_class}


def _summary(cov: dict) -> str:
    t = cov["totals"]
    visits = sum(v for k, v in t.items() if k.startswith("visits_") and k not in ("visits_no_choice", "visits_other"))
    lines = [f"visits with a recorded option: {visits}; no option choice: {t.get('visits_no_choice', 0)}; events of other mods: {t.get('visits_other', 0)}"]
    for c in CLASSES:
        lines.append(f"  {c:11s} visits {t.get('visits_' + c, 0):6d} ({100 * t.get('visits_' + c, 0) / max(visits, 1):5.1f}%)   "
                     f"decisions {t.get('decisions_' + c, 0):6d}")
    cls = Counter(cov["event_class"].values())
    lines.append("events by weakest key (keys chosen at least %d times): %s" % (MIN_CELL, dict(cls)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("event")
    r.add_argument("title")
    r.add_argument("--offered", nargs="*", default=None)
    c = sub.add_parser("coverage")
    c.add_argument("--counts", required=True)
    c.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    tab = table()
    if a.cmd == "resolve":
        print(tab.resolve(a.event, a.title, {"offered": a.offered} if a.offered else None))
        return 0
    doc = json.loads(Path(a.counts).read_text(encoding="utf-8"))
    cov = coverage(tab, doc["seqs"], doc.get("visits"))
    print(json.dumps(cov, indent=1) if a.json else _summary(cov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
