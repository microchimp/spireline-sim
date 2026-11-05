#!/usr/bin/env python3
"""Build `data/event_option_titles.json`: for each event, its option pages with the option keys the human run record uses, the index each
option has on the page (the copy's action number), the game's displayed title, and where each fact came from.

    python scripts/build_event_option_titles.py scan  --out COUNTS.json      # reads the human corpus once (aggregates only)
    python scripts/build_event_option_titles.py build --counts COUNTS.json   # writes data/event_option_titles.json
    python scripts/build_event_option_titles.py check                        # the file agrees with the captures

The human record (Spire Codex public run export for v0.107.1) names the option a player took by a text key,
`<EVENT>.pages.<PAGE>.options.<OPTION>.title`; the copy takes an option by its index on the page. The two sources that join them:

  * the live captures in `tests/fixtures/events/` (the game's own pages: `options[].index` and `title`, and the page that follows an
    option in the `-opt<n>` files). The record's option name is matched to a captured title by its words (`TOUCH_A_MIRROR` is
    "Touch a Mirror"; `RIP` is "Rip the Leech Off"); a name whose words are all in exactly one title of the page is `exact`.
  * the engine's own event code (`src/Sts2Emulator/Core/Run/RunEngine.cs`: which action number each option is, the lock rules, the
    order) and the record's outcomes (gold, HP, cards, relics each option led to in single-choice visits), for names no title carries
    (`PLAIN` and `ORNATE` for "This" and "That"). These are `inferred`, with the evidence written in the row.

What neither source gives stays `unresolved` and goes on the owner's list: pages whose options are rolled per visit (the order of the
offered dolls, the Tinker Time riders, Colorful Philosophers' pools, the trades) have an index only when the caller says which options
are on the page (`state["offered"]`). The decompile is not used and not needed; `scripts/generate_event_options.py` is not run.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "events"
CONSTANTS = ROOT / "src" / "Sts2Emulator" / "Core" / "Run" / "RunConstants.cs"
OUT = ROOT / "data" / "event_option_titles.json"
RELEASE = "v0.107.1"
RECORD = "Spire Codex public run export for v0.107.1"
MIN_CELL = 20  # D13: a count over fewer visits is not shown
KEY = re.compile(r"^([A-Z0-9_]+)\.pages\.([A-Z0-9_]+)\.options\.([A-Z0-9_]+)\.title$")

# ----------------------------------------------------------------------------------------------------------------------- curated rows
# (event, page, key, index, source, mapping, evidence). `source`: capture | engine | record-inferred. `mapping`: exact | inferred.
# A key written `re:<pattern>` is a pattern (the pages of Slippery Bridge and Tablet of Truth carry a count in their names).
# Rows for keys no captured title carries. Everything not listed here is matched to a captured title by its words.
ROWS: list[tuple] = [
    ("LOST_WISP", "INITIAL", "CLAIM", 0, "record-inferred", "inferred",
     "capture index 0 is 'Capture the Wisp' (adds Decay, gives the Lost Wisp); in the record CLAIM leaves one card and one relic in every visit, SEARCH about 60 gold"),
    ("LUMINOUS_CHOIR", "INITIAL", "OFFER_TRIBUTE", 1, "engine", "inferred",
     "capture index 1 is the option locked by 'Requires N Gold'; the engine opens action 1 on the tribute price (RunEngine event mask); the record shows a relic taken"),
    ("RANWID_THE_ELDER", "INITIAL", "POTION", 0, "capture", "inferred",
     "capture index 0 is locked with 'no Potions that can be given'; the engine mask orders potion, gold, relic"),
    ("RANWID_THE_ELDER", "INITIAL", "RELIC", 2, "capture", "inferred",
     "capture index 2 is locked with 'no Relics that can be given'"),
    ("SAPPHIRE_SEED", "INITIAL", "EAT", 0, "record-inferred", "inferred",
     "capture index 0 is 'Consume' (heal 9, upgrade a card); in the record EAT heals about 9 and upgrades one card in every visit, PLANT enchants a card"),
    ("SELF_HELP_BOOK", "INITIAL", "READ_ENTIRE_BOOK", 2, "capture", "inferred",
     "capture index 2 is locked with 'no Powers that can be Enchanted'; the engine's pages are Attack 0, Skill 1, Power 2; READ_THE_BACK and READ_PASSAGE take 0 and 1"),
    ("STONE_OF_ALL_TIME", "INITIAL", "LIFT", 0, "capture", "inferred",
     "capture index 0 is locked with 'Requires a Potion'; in the record LIFT heals about 10 and raises max HP by 10"),
    ("TEA_MASTER", "INITIAL", "EMBER_TEA", 1, "capture", "inferred",
     "capture index 1 is locked with 'Requires 150 Gold'; the engine's price order is Bone Tea 0, Ember Tea 1, Tea of Discourtesy 2"),
    ("THIS_OR_THAT", "INITIAL", "PLAIN", 0, "record-inferred", "inferred",
     "capture index 0 is 'This' (lose 6 HP, gain gold); in the record PLAIN costs 6 HP and pays about 55 gold in every visit"),
    ("THIS_OR_THAT", "INITIAL", "ORNATE", 1, "record-inferred", "inferred",
     "capture index 1 is 'That' (add Clumsy, obtain a random relic); in the record ORNATE adds one card and one relic in every visit"),
    ("TINKER_TIME", "INITIAL", "CHOOSE_CARD_TYPE", 0, "engine", "inferred",
     "the capture shows one option, 'Accept', which only turns the page; the record has one key on this page"),
    ("WELCOME_TO_WONGOS", "INITIAL", "BARGAIN_BIN", 0, "capture", "inferred",
     "capture indices 0 to 2 are locked at 'Requires 100 / 200 / 300 Gold'; the engine's price order is bargain bin, featured item, mystery box"),
    ("WELCOME_TO_WONGOS", "INITIAL", "FEATURED_ITEM", 1, "capture", "inferred", "second price tag (200 gold), see BARGAIN_BIN"),
    ("WELCOME_TO_WONGOS", "INITIAL", "MYSTERY_BOX", 2, "capture", "inferred", "third price tag (300 gold), see BARGAIN_BIN"),
    ("ZEN_WEAVER", "INITIAL", "EMOTIONAL_AWARENESS", 1, "capture", "inferred",
     "capture indices 1 and 2 are locked 'Not enough Gold'; the engine's order is breathing 0, emotional 1, acupuncture 2; in the record EMOTIONAL removes one card, ACUPUNCTURE two"),
    ("ZEN_WEAVER", "INITIAL", "ARACHNID_ACUPUNCTURE", 2, "capture", "inferred", "see EMOTIONAL_AWARENESS"),
    # --- positional names: the trader lays its offers out in order
    ("RELIC_TRADER", "INITIAL", "TOP", 0, "record-inferred", "inferred",
     "the offers (up to three of the player's relics, shuffled) are named by their place on the page, top to bottom"),
    ("RELIC_TRADER", "INITIAL", "MIDDLE", 1, "record-inferred", "inferred", "second place, see TOP"),
    ("RELIC_TRADER", "INITIAL", "BOTTOM", 2, "record-inferred", "inferred", "third place, see TOP"),
    # --- later pages the captures show (the option list after the choice)
    ("ABYSSAL_BATHS", "ALL", "LINGER", 0, "capture", "exact", "capture page after Immerse: Linger 0, Exit Baths 1"),
    ("ABYSSAL_BATHS", "ALL", "EXIT_BATHS", 1, "capture", "exact", "capture page after Immerse"),
    ("SLIPPERY_BRIDGE", "INITIAL", "OVERCOME", 0, "capture", "exact",
     "capture page after Hold On: Overcome 0, Hold On 1; the record keeps the INITIAL page name for Overcome on every later page"),
    ("SLIPPERY_BRIDGE", "re:HOLD_ON_(?:\\d+|LOOP)", "re:HOLD_ON_(?:\\d+|LOOP)", 1, "capture", "exact",
     "capture page after Hold On: Hold On is index 1; the record counts the holds in the names"),
    ("TABLET_OF_TRUTH", "re:DECIPHER(?:_\\d+)?", "DECIPHER", 0, "capture", "inferred",
     "capture page after Decipher: 'Continue Deciphering' 0, 'Give Up' 1"),
    ("TABLET_OF_TRUTH", "re:DECIPHER(?:_\\d+)?", "GIVE_UP", 1, "capture", "exact", "capture page after Decipher"),
    ("DENSE_VEGETATION", "REST", "FIGHT", 0, "engine", "inferred",
     "the engine's Rest page offers the fight only (action 0); the capture of that page was taken before it changed and still shows the first page"),
    ("PUNCH_OFF", "I_CAN_TAKE_THEM", "FIGHT", 0, "capture", "exact", "capture page after 'I Can Take Them': Fight 0"),
    ("THE_LANTERN_KEY", "KEEP_THE_KEY", "FIGHT", 0, "capture", "exact", "capture page after 'Keep the Key': Fight 0"),
    ("TRIAL", "re:MERCHANT|NOBLE|NONDESCRIPT", "GUILTY", 0, "capture", "exact", "capture page after Accept: 'DECIDE: Guilty' 0, 'DECIDE: Innocent' 1"),
    ("TRIAL", "re:MERCHANT|NOBLE|NONDESCRIPT", "INNOCENT", 1, "capture", "exact", "capture page after Accept"),
    ("TRIAL", "REJECT", "ACCEPT", 0, "capture", "exact", "capture page after Reject: Accept 0, Double Down 1"),
    ("TRIAL", "REJECT", "DOUBLE_DOWN", 1, "capture", "exact", "capture page after Reject"),
    ("ENDLESS_CONVEYOR", "INITIAL", "OBSERVE_CHEF", 1, "capture", "exact", "capture page 1: 'Observe the Chef' 1"),
    ("ENDLESS_CONVEYOR", "ALL", "FRIED_EEL", 0, "capture", "exact", "capture page 1: 'Grab Fried Eel off the Belt' 0"),
    ("ENDLESS_CONVEYOR", "ALL", "JELLY_LIVER", 0, "capture", "exact", "capture page after Grab: 'Grab Jelly Liver off the Belt' 0"),
    ("ENDLESS_CONVEYOR", "GRAB_SOMETHING_OFF_THE_BELT", "LEAVE", 1, "capture", "exact", "capture page after Grab: Leave 1"),
    ("WAR_HISTORIAN_REPY", "INITIAL", "UNLOCK_CHEST", 1, "capture", "exact", "capture 'Unlock the Chest' 1 (a second page offers only the door not yet opened)"),
    # --- engine only: events with no capture
    ("GRAVE_OF_THE_FORGOTTEN", "INITIAL", "CONFRONT", 0, "engine", "inferred",
     "engine action 0 is Confront (adds Decay, then a card to enchant); in the record CONFRONT adds one card and enchants one in every visit"),
    ("GRAVE_OF_THE_FORGOTTEN", "INITIAL", "ACCEPT", 1, "engine", "inferred", "engine action 1 takes Forgotten Soul; the record shows one relic and no card"),
    ("ROUND_TEA_PARTY", "INITIAL", "ENJOY_TEA", 0, "engine", "inferred",
     "engine action 0 takes Royal Poison and heals to full; in the record ENJOY_TEA heals about 45 and gives one relic"),
    ("ROUND_TEA_PARTY", "INITIAL", "PICK_FIGHT", 1, "engine", "inferred",
     "engine action 1 opens the warning page whose one option charges 11 HP for a relic; the record shows about 11 damage and one relic"),
    ("ROUND_TEA_PARTY", "CONTINUE_FIGHT", "CONTINUE_FIGHT", 0, "engine", "inferred", "the warning page's one option; the record does not list it"),
    ("COLOSSAL_FLOWER", "INITIAL", "EXTRACT_CURRENT_PRIZE_1", 0, "engine", "inferred",
     "engine action 0 pays the rung's gold (35, 75, 135); in the record this option pays 35 gold in every visit"),
    ("COLOSSAL_FLOWER", "INITIAL", "REACH_DEEPER_1", 1, "engine", "inferred", "engine action 1 digs deeper for 5, 6, 7 damage"),
    ("COLOSSAL_FLOWER", "REACH_DEEPER_1", "EXTRACT_CURRENT_PRIZE_2", 0, "engine", "inferred", "second rung, see above"),
    ("COLOSSAL_FLOWER", "REACH_DEEPER_1", "REACH_DEEPER_2", 1, "engine", "inferred", "second rung, see above"),
    ("COLOSSAL_FLOWER", "REACH_DEEPER_2", "EXTRACT_INSTEAD", 0, "engine", "inferred", "third rung: action 0 extracts for 135, action 1 takes Pollinous Core"),
    ("COLOSSAL_FLOWER", "REACH_DEEPER_2", "POLLINOUS_CORE", 1, "engine", "inferred", "third rung, see above"),
]
# Ender conveyor dishes other than the two the captures show: the grab is always option 0 of its page.
for _dish in ("CAVIAR", "CLAM_ROLL", "GOLDEN_FYSH", "SEAPUNK_SALAD", "SPICY_SNAPPY", "SUSPICIOUS_CONDIMENT"):
    ROWS.append(("ENDLESS_CONVEYOR", "ALL", _dish, 0, "capture", "inferred",
                 "the belt's grab is option 0 whatever dish it carries (captured for Fried Eel and Jelly Liver, title 'Grab <dish> off the Belt')"))

# Pages whose options are rolled per visit: (event, page, key, reason). Their index needs `state["offered"]` (the options on the page,
# in display order); without it the key is unresolved.
DYNAMIC: list[tuple] = [
    ("TINKER_TIME", "CHOOSE_CARD_TYPE", "ATTACK", "two of Attack, Skill, Power are offered, in a rolled order (engine EventRandomOffer)"),
    ("TINKER_TIME", "CHOOSE_CARD_TYPE", "SKILL", "see ATTACK"),
    ("TINKER_TIME", "CHOOSE_CARD_TYPE", "POWER", "see ATTACK"),
    *[("TINKER_TIME", "CHOOSE_RIDER", r, "two riders of the chosen type are offered, in a rolled order")
      for r in ("CHAOS", "CHOKING", "CURIOUS", "ENERGIZED", "EXPERTISE", "IMPROVEMENT", "SAPPING", "VIOLENCE", "WISDOM")],
    *[("COLORFUL_PHILOSOPHERS", "INITIAL", c, "one option per other character's pool, at most three, in a rolled order")
      for c in ("SILENT", "DEFECT", "NECROBINDER", "REGENT")],
    ("THE_FUTURE_OF_POTIONS", "INITIAL", "POTION", "one option per potion held (up to three), shuffled; the record does not say which potion"),
    ("DOLL_ROOM", "DOLLS", "DAUGHTER_OF_THE_WIND", "the offered dolls (two or all three) are a shuffle from the event's own stream"),
    ("DOLL_ROOM", "DOLLS", "MR_STRUGGLES", "see DAUGHTER_OF_THE_WIND"),
    ("DOLL_ROOM", "DOLLS", "BING_BONG", "see DAUGHTER_OF_THE_WIND"),
]
# The displayed title of the curated rows on pages after the first (captured pages: the option list after the choice).
TITLES: dict[tuple, str] = {
    ("ABYSSAL_BATHS", "ALL", "LINGER"): "Linger",
    ("ABYSSAL_BATHS", "ALL", "EXIT_BATHS"): "Exit Baths",
    ("SLIPPERY_BRIDGE", "re:HOLD_ON_(?:\\d+|LOOP)", "re:HOLD_ON_(?:\\d+|LOOP)"): "Hold On",
    ("TABLET_OF_TRUTH", "re:DECIPHER(?:_\\d+)?", "DECIPHER"): "Continue Deciphering",
    ("TABLET_OF_TRUTH", "re:DECIPHER(?:_\\d+)?", "GIVE_UP"): "Give Up",
    ("PUNCH_OFF", "I_CAN_TAKE_THEM", "FIGHT"): "Fight",
    ("THE_LANTERN_KEY", "KEEP_THE_KEY", "FIGHT"): "Fight",
    ("TRIAL", "re:MERCHANT|NOBLE|NONDESCRIPT", "GUILTY"): "DECIDE: Guilty",
    ("TRIAL", "re:MERCHANT|NOBLE|NONDESCRIPT", "INNOCENT"): "DECIDE: Innocent",
    ("TRIAL", "REJECT", "ACCEPT"): "Accept",
    ("TRIAL", "REJECT", "DOUBLE_DOWN"): "Double Down",
    ("ENDLESS_CONVEYOR", "GRAB_SOMETHING_OFF_THE_BELT", "LEAVE"): "Leave",
    ("ENDLESS_CONVEYOR", "ALL", "FRIED_EEL"): "Grab Fried Eel off the Belt",
    ("ENDLESS_CONVEYOR", "ALL", "JELLY_LIVER"): "Grab Jelly Liver off the Belt",
}
# Titles that carry a changing word are matched by a pattern as well (digits are patterned by the builder).
TITLE_PATTERNS: dict[tuple, str] = {("ENDLESS_CONVEYOR", "ALL"): r"Grab .+ off the Belt"}
# Options the game takes as the continuation of a choice and the record does not list: (event, page, key) -> the key they follow.
UNRECORDED: dict[tuple, str] = {("ROUND_TEA_PARTY", "CONTINUE_FIGHT", "CONTINUE_FIGHT"): "PICK_FIGHT"}

# Names the record uses that the table does not resolve at all (a count of one or two visits, or a name the game's page never offers).
NOT_RESOLVED: list[tuple] = [
    ("REFLECTIONS", "INITIAL", "STARE", "one visit in the record; no capture or engine option of that name"),
    ("COLORFUL_PHILOSOPHERS", "INITIAL", "IRONCLAD", "one visit; the Ironclad's own pool is not offered to an Ironclad"),
]
# Events the record lists with no option key at all (their screen is not an option page) and events of other mods.
NO_CHOICE_EVENTS = {"FAKE_MERCHANT": "the stall is its own screen; the record keeps the purchases, not an option"}
STOP_WORDS = {"THE", "A", "AN", "OF", "TO", "OFF", "INTO", "IN", "ON", "YOUR", "AND"}


# --------------------------------------------------------------------------------------------------------------------------- captures
def slug(name: str) -> str:
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[A-Z])", "_", name.strip())
    return re.sub(r"[^A-Z0-9_]", "", re.sub(r"\s+", "_", text.upper()))


def engine_ids() -> dict[str, int]:
    """Game event id (`ABYSSAL_BATHS`) -> the copy's event id, read off RunConstants.cs's `Event<Name> = n` lines."""
    text = CONSTANTS.read_text(encoding="utf-8")
    skip = {"SKIP_ACTION"}  # `EventSkipAction = 3` is the proceed action, not an event
    return {slug(name): int(v) for name, v in re.findall(r"public const int Event(\w+) = (\d+);", text) if int(v) > 0 and slug(name) not in skip}


def words(text: str) -> list[str]:
    return re.sub(r"[^A-Za-z0-9]+", " ", text).upper().split()


def load_captures() -> dict[str, dict]:
    """event id -> {"page1": [options], "after": {chosen index: [options of the next page]}, "files": [...]}.

    The first page is read from the `-opt<n>` files' `options` (the page the choice was made on), and from `-options.json` when it
    agrees: `JungleMazeAdventure-options.json` holds another event's page (Endless Conveyor's), the `-opt` files hold the right one."""
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(FIXTURES / "*.json"))):
        name = Path(path).name
        if "-sphere-" in name:
            continue
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        eid = doc.get("event_id")
        if not eid:
            continue
        rec = out.setdefault(eid, {"page1": None, "page1_files": [], "after": {}, "files": []})
        rec["files"].append(name)
        opts = [dict(index=o["index"], title=o["title"], description=o.get("description", ""), is_locked=bool(o.get("is_locked")),
                     is_proceed=bool(o.get("is_proceed"))) for o in doc.get("options") or []]
        if "chosen" in doc:
            rec.setdefault("page1_by_opt", []).append(opts)
            ev = (doc.get("after") or {}).get("event") or {}
            nxt = [dict(index=o["index"], title=o["title"], description=o.get("description", ""), is_locked=bool(o.get("is_locked")),
                        is_proceed=bool(o.get("is_proceed"))) for o in ev.get("options") or []]
            rec["after"][int(doc["chosen"])] = {"state": doc.get("after_state_type"), "options": nxt}
        else:
            rec["options_file"] = opts
    for rec in out.values():
        by_opt = rec.get("page1_by_opt") or []
        # the first page: the `-opt` files agree with each other; the plain options file is the same page unless it is stale
        page1 = by_opt[0] if by_opt else rec.get("options_file")
        if by_opt and rec.get("options_file") and [(o["index"], o["title"]) for o in by_opt[0]] != [(o["index"], o["title"]) for o in rec["options_file"]]:
            rec["options_file_stale"] = True
        rec["page1"] = page1
    return out


def match_title(key: str, page: list[dict]) -> tuple[int | None, str]:
    """The index of the captured option whose title the record's option name spells, and how: 'equal' (the same words), 'words' (the
    name's words, less a count, are all in exactly one title) or '' (no single match)."""
    flat = "_".join(words(key))
    cands = [o for o in page if not o["is_locked"] or o["title"] != "Locked"]
    eq = [o["index"] for o in cands if "_".join(words(o["title"])) == flat]
    if len(eq) == 1:
        return eq[0], "equal"
    stripped = re.sub(r"_\d+$", "", flat)
    if stripped != flat:
        eq = [o["index"] for o in cands if "_".join(words(o["title"])) == stripped]
        if len(eq) == 1:
            return eq[0], "equal"
    kw = {w for w in words(flat.replace("_", " ")) if not w.isdigit()}
    hits = [o["index"] for o in cands if kw and kw <= set(words(o["title"]))]
    if len(hits) == 1:
        return hits[0], "words"
    return None, ""


# ------------------------------------------------------------------------------------------------------------------------------ scan
def cmd_scan(a: argparse.Namespace) -> int:
    """One pass over the corpus: the event visits of the training fifths (the exam fifth is left out with `human_starts.is_exam`) and
    which option keys they chose, in order. Aggregates only; nothing per run is written."""
    sys.path.insert(0, str(ROOT / "rl"))
    sys.path.insert(0, str(ROOT / "src"))
    import human_starts as hs  # noqa: PLC0415

    ids = hs.read_exam_ids()
    stat: Counter = Counter()
    seqs: Counter = Counter()
    visits: Counter = Counter()
    bare: Counter = Counter()
    t0 = time.time()
    with gzip.open(a.corpus or hs.CORPUS, "rt", encoding="utf-8") as fh:
        for line in fh:
            stat["lines"] += 1
            if '"ascension":10,' not in line or '"character":"CHARACTER.IRONCLAD"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not hs.human_ok(r) or not hs.vanilla_acts(r):
                continue
            if hs.is_exam(hs.run_id(r), ids):
                stat["exam_left_out"] += 1
                continue
            stat["runs"] += 1
            for act in r.get("map_point_history") or []:
                for p in act:
                    mid = str((p.get("rooms") or [{}])[0].get("model_id") or "")
                    if not mid.startswith("EVENT.") or p.get("map_point_type") == "ancient" or mid == "EVENT.NEOW":
                        continue
                    ev = mid.split(".", 1)[1]
                    ps = (p.get("player_stats") or [{}])[0]
                    visits[ev] += 1
                    keys = []
                    for c in ps.get("event_choices") or []:
                        k = str((c.get("title") or {}).get("key") or "")
                        m = KEY.match(k)
                        if m:
                            keys.append(f"{m.group(2)}.{m.group(3)}")
                        else:
                            keys.append("~" + k.removesuffix(".title"))
                            bare[(ev, k.removesuffix(".title"))] += 1
                    seqs[(ev, tuple(keys))] += 1
    doc = {"record": RECORD, "release": RELEASE, "stat": dict(stat), "visits": dict(visits),
           "seqs": [[ev, list(keys), n] for (ev, keys), n in seqs.items()], "seconds": round(time.time() - t0)}
    Path(a.out).write_text(json.dumps(doc), encoding="utf-8")
    print("scanned", dict(stat), "event visits", sum(visits.values()), "seconds", doc["seconds"])
    return 0


# ----------------------------------------------------------------------------------------------------------------------------- build
def _suppress(n: int):
    return n if n >= MIN_CELL else f"<{MIN_CELL}"


def build(counts: dict | None) -> dict:
    caps = load_captures()
    ids = engine_ids()
    n_key: Counter = Counter()  # (event, page, key) -> choices in the record
    visits_of: Counter = Counter()
    if counts:
        for ev, keys, n in counts["seqs"]:
            for k in keys:
                if k.startswith("~"):
                    if ".pages." in k:
                        continue  # an option key of another mod's content (a card pool the game does not have): not a table row
                    page, key = "DOLLS", k[1:]
                else:
                    page, key = k.split(".", 1)
                n_key[(ev, page, key)] += n
        visits_of.update(counts["visits"])
    rows_by_event: dict[str, list[tuple]] = defaultdict(list)
    for r in ROWS:
        rows_by_event[r[0]].append(r)
    events: dict[str, dict] = {}

    def cap_title(page1, idx):
        o = next((o for o in page1 if o["index"] == idx), None)
        return o["title"] if o and o["title"] not in ("Locked", "Proceed") and not o["is_proceed"] else None

    def row(index, key, title, source, mapping, evidence, *, locked=False, proceed=False, key_re=None, n=None, desc=None):
        out = {"index": index, "key": key, "title": title, "is_proceed": proceed, "is_locked": locked, "source": source, "mapping": mapping}
        if key_re:
            out["key_pattern"] = key_re
        if title and re.search(r"\d", title):
            out["title_pattern"] = re.sub(r"\d+", lambda _m: r"\d+", re.escape(title))
        if n is not None:
            out["n"] = _suppress(n)
        if evidence:
            out["evidence"] = evidence
        if desc:
            out["description"] = desc
        return out

    def count(ev, page, key, page_re=False, key_re=False):
        total = 0
        for (e, pg, k), n in n_key.items():
            if e != ev:
                continue
            if (re.fullmatch(page[3:], pg) if page_re else pg == page) and (re.fullmatch(key[3:], k) if key_re else k == key):
                total += n
        return total

    names = sorted(set(caps) | set(rows_by_event) | {e for (e, _, _) in n_key} | {d[0] for d in DYNAMIC} | set(visits_of))
    modded = {e for e in names if "-" in e or e.startswith(("STS2", "YUI", "BS_", "XUELI", "LEXNINJA", "ZSPROJECT", "ACTSFROMTHEPAST", "INTEGRATED"))}
    for ev in names:
        if ev in modded or ev not in ids:
            continue
        cap = caps.get(ev)
        pages: list[dict] = []
        used: set = set()
        # --- page INITIAL: the captured options; names matched to titles by their words, then the curated rows
        page1 = (cap or {}).get("page1") or []
        keys_here = sorted({k for (e, pg, k) in n_key if e == ev and pg == "INITIAL"})
        rows: list[dict] = []
        cur = {r[2]: r for r in rows_by_event.get(ev, []) if r[1] == "INITIAL"}
        matched: dict[int, list[str]] = defaultdict(list)
        for key in keys_here:
            if key in cur:
                continue
            idx, how = match_title(key, page1)
            if idx is not None:
                matched[idx].append(key)
        for key in keys_here:
            if key in cur:
                _, _, _, idx, src, mapping, evidence = cur[key]
                title = cap_title(page1, idx)
                opt = next((o for o in page1 if o["index"] == idx), None)
                rows.append(row(idx, key, title, src, mapping, evidence, locked=bool(opt and opt["is_locked"]), n=count(ev, "INITIAL", key),
                                key_re=None))
                used.add(key)
                continue
            idx, how = match_title(key, page1)
            if idx is None or len(matched[idx]) != 1:
                continue
            opt = next(o for o in page1 if o["index"] == idx)
            re_key = None
            if re.search(r"_\d+$", key) and "_".join(words(opt["title"])) != "_".join(words(key)):  # a count in the name the title lacks
                re_key = re.sub(r"_\d+$", r"_\\d+", key)
            rows.append(row(idx, key, opt["title"], "capture", "exact", f"words of the name are in captured title {opt['title']!r} ({how})",
                            key_re=re_key, n=count(ev, "INITIAL", key), desc=opt["description"] or None))
            used.add(key)
        for r in rows_by_event.get(ev, []):
            if r[1] != "INITIAL" or r[2] in used:
                continue
            # a curated INITIAL row for a name the record never used in this sample: kept (the key may occur in the held-out part)
            _, _, key, idx, src, mapping, evidence = r
            opt = next((o for o in page1 if o["index"] == idx), None)
            rows.append(row(idx, key, cap_title(page1, idx), src, mapping, evidence,
                            locked=bool(opt and opt["is_locked"]), n=count(ev, "INITIAL", key)))
            used.add(key)
        # options of the captured page no record name was matched to (nothing in the training fifths chose them, or they are locked
        # with no name): listed so the page is complete
        have = {r["index"] for r in rows}
        for o in page1:
            if o["index"] not in have and not o["is_proceed"] and o["title"] != "Proceed":
                rows.append(row(o["index"], None, None if o["title"] == "Locked" else o["title"], "capture", "unresolved",
                                "no record name is matched to this captured option", locked=o["is_locked"], proceed=o["is_proceed"],
                                desc=o["description"] or None))
        rows.sort(key=lambda r: (r["index"], r["key"] or ""))
        if rows:
            pages.append({"page": "INITIAL", "page_pattern": False, "options": rows})
        # --- other pages: curated rows grouped by page
        by_page: dict[str, list[tuple]] = defaultdict(list)
        for r in rows_by_event.get(ev, []):
            if r[1] != "INITIAL":
                by_page[r[1]].append(r)
        for pg, rs in by_page.items():
            is_re = pg.startswith("re:")
            prow = []
            for _, _, key, idx, src, mapping, evidence in rs:
                key_is_re = key.startswith("re:")
                title = TITLES.get((ev, pg, key))
                prow.append(row(idx, key[3:] if key_is_re else key, title, src, mapping, evidence,
                                key_re=key[3:] if key_is_re else None,
                                n=count(ev, pg, key, page_re=is_re, key_re=key_is_re)))
                if (ev, pg, key) in UNRECORDED:
                    prow[-1]["recorded"] = False
                    prow[-1]["after_key"] = UNRECORDED[(ev, pg, key)]
            prow.sort(key=lambda r: r["index"])
            if (ev, pg) in TITLE_PATTERNS:
                for r in prow:
                    r["title_pattern"] = TITLE_PATTERNS[(ev, pg)]
            pages.append({"page": pg[3:] if is_re else pg, "page_pattern": is_re, "options": prow})
        # --- dynamic pages
        dyn_by_page: dict[str, list[tuple]] = defaultdict(list)
        for d in DYNAMIC:
            if d[0] == ev:
                dyn_by_page[d[1]].append(d)
        for pg, ds in dyn_by_page.items():
            prow = []
            for _, _, key, reason in ds:
                r = row(None, key, None, "engine", "unresolved", reason, n=count(ev, pg, key))
                r["dynamic"] = True
                prow.append(r)
            pages.append({"page": pg, "page_pattern": False, "dynamic": True, "options": prow})
        # --- names the table does not resolve
        for nr in NOT_RESOLVED:
            if nr[0] == ev:
                for p in pages:
                    if p["page"] == nr[1]:
                        p["options"].append(row(None, nr[2], None, "record-inferred", "unresolved", nr[3], n=count(ev, nr[1], nr[2])))
                        break
                else:
                    pages.append({"page": nr[1], "page_pattern": False,
                                  "options": [row(None, nr[2], None, "record-inferred", "unresolved", nr[3], n=count(ev, nr[1], nr[2]))]})
        # --- record names that fell through (not matched, not curated, not dynamic): unresolved rows so the gap is visible
        seen_keys = {(p["page"], r["key"]) for p in pages for r in p["options"] if r["key"]}
        for (e, pg, k), n in sorted(n_key.items()):
            if e != ev:
                continue
            if any(p["page_pattern"] and re.fullmatch(p["page"], pg) for p in pages) or any(
                    r.get("key_pattern") and re.fullmatch(r["key_pattern"], k) for p in pages for r in p["options"]):
                continue
            if (pg, k) in seen_keys:
                continue
            target = next((p for p in pages if p["page"] == pg), None)
            if target is None:
                target = {"page": pg, "page_pattern": False, "options": []}
                pages.append(target)
            target["options"].append(row(None, k, None, "record-inferred", "unresolved", "no captured title or engine option is matched to this name",
                                         n=n))
        status_src = {r["source"] for p in pages for r in p["options"] if r["mapping"] != "unresolved"}
        ev_doc = {"engine_id": ids[ev], "captured": cap is not None, "pages": pages}
        if cap and cap.get("options_file_stale"):
            ev_doc["note"] = "the plain options capture holds another event's page; the first page is read from the -opt captures"
        if ev in NO_CHOICE_EVENTS:
            ev_doc["no_option_choice"] = NO_CHOICE_EVENTS[ev]
        if ev in visits_of:
            ev_doc["visits"] = _suppress(visits_of[ev])
        ev_doc["sources"] = sorted(status_src)
        events[ev] = ev_doc
    return {"record": RECORD, "release": RELEASE,
            "_comment": ("Event option pages: the human run record's option names (Spire Codex public run export for v0.107.1) to the copy's "
                         "option index. Built by scripts/build_event_option_titles.py from the live captures in tests/fixtures/events, the "
                         "engine's event code and the record's own keys and outcomes (aggregates; counts under 20 are not shown). "
                         "mapping: exact = the name's words are in a captured title of that page; inferred = engine order or outcomes, "
                         "evidence in the row; unresolved = no index (dynamic = the index depends on the options the visit offered)."),
            "events": events}


def cmd_build(a: argparse.Namespace) -> int:
    counts = json.loads(Path(a.counts).read_text(encoding="utf-8")) if a.counts else None
    doc = build(counts)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    n_opts = sum(len(p["options"]) for e in doc["events"].values() for p in e["pages"])
    print("wrote", a.out, "events", len(doc["events"]), "option rows", n_opts)
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    """Every captured first-page option of the file has the captured index and title."""
    doc = json.loads(Path(a.out).read_text(encoding="utf-8"))
    caps = load_captures()
    bad = 0
    for ev, cap in caps.items():
        if ev not in doc["events"]:
            print("missing event", ev)
            bad += 1
            continue
        page = next((p for p in doc["events"][ev]["pages"] if p["page"] == "INITIAL"), None)
        for o in [o for o in (cap["page1"] or []) if not o["is_proceed"]]:
            hit = [r for r in (page or {"options": []})["options"] if r["index"] == o["index"]]
            if not hit:
                print("no row for", ev, o["index"], o["title"])
                bad += 1
    print("check", "ok" if not bad else f"{bad} problems")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--out", required=True)
    s.add_argument("--corpus", default=None)
    s.set_defaults(fn=cmd_scan)
    b = sub.add_parser("build")
    b.add_argument("--counts", default=None)
    b.add_argument("--out", default=str(OUT))
    b.set_defaults(fn=cmd_build)
    c = sub.add_parser("check")
    c.add_argument("--out", default=str(OUT))
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
