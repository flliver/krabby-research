#!/usr/bin/env python3
"""The candidate ladder's ledger reader and user-override CLI (omni task, 2026-09-19).

The ladder driver (``run_candidates.py``) owns ``ledger.json`` and never reads a user's intent from
anywhere but ``ledger_overrides.jsonl``, which is what this tool appends to. The split is the whole
point: the driver may be inside a two-hour training window when the user answers a notification, so
an answer has to be a durable, append-only, uuid-stamped fact that the running driver picks up at
its next ``ingest_overrides()`` (within one poll tick) and that a STOPPED driver picks up at its
next launch. Nothing here ever writes ``ledger.json`` or ``state.json``.

    ledger_cli.py --campaign-dir <dir> pending
    ledger_cli.py --campaign-dir <dir> show [<id>] [--json]
    ledger_cli.py --campaign-dir <dir> confirm <Did> [--why ...]
    ledger_cli.py --campaign-dir <dir> overrule <Did> --choose ADOPT|REJECT|SKIP|PASS|WATCH|FAIL --why ...
    ledger_cli.py --campaign-dir <dir> decide <Cid> --choose run|skip [--why ...]
    ledger_cli.py --campaign-dir <dir> resume-milestone <Mid> (--accept | --drop <Cid>) [--why ...]
    ledger_cli.py --campaign-dir <dir> stop | unstop

This module is also the notification CONTRACT: :func:`format_notify` writes every line the driver
puts in ``notify.log`` and :func:`parse_notify` reads one back (round-trip tested), so the
assistant's Monitor can relay a line and quote the command it carries verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid as _uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: campaign -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
REPO = HERE.parents[5]

LEDGER_NAME = "ledger.json"
OVERRIDES_NAME = "ledger_overrides.jsonl"
STATE_NAME = "state.json"
STOP_NAME = "STOP"

#: Every tag the driver may emit. The assistant's Monitor relays ALL of them. ``MIGRATION``
#: (per-category plan 2026-09-21, B5) carries the eval-only rescreen / reeval / rescore items that
#: move a paused campaign onto a new ``eval_sha``; its lines say "eval-only" (nothing trains).
NOTIFY_TAGS = ("RESOLUTION", "DECISION", "CONFIRM", "SELF-OVERRULE", "OVERRIDE-APPLIED", "SQUASH",
               "MILESTONE", "RETENTION", "INFRA", "HALT", "DONE", "RELAUNCH", "AWAIT", "MIGRATION")

#: What ``show`` prints for an entry (or a campaign) recorded before ``eval_sha`` existed (D8):
#: every pre-suite record -- D00..D07 of the 2026-09-19 ladder -- has none.
PREDATES_SUITE = "— (predates the suite)"

#: Milestone entries. They have their OWN verb (``resume-milestone``) and are not decisions.
MILESTONE_KINDS = ("milestone", "milestone_s2")

#: Entry kinds a user may never confirm or overrule. Two reasons, both fatal in their own way:
#: a ``resolution``/``measurement``/``baseline`` is a MEASUREMENT, not a choice -- overruling one
#: would let a wish rewrite the resolution every decision is standardised by; and a ``milestone``
#: is answered with ``resume-milestone``, not with ``confirm``/``overrule`` (an ``overrule M1
#: --choose FAIL`` moved the closed entry to ``overruled``, which sent ``_next_item_raw`` back to
#: the OPEN branch, re-booked the 4.3-hour promotion, and then died in ``_close_entry`` on an
#: illegal ``overruled -> final`` transition with no HALT line and no phase).
UNOVERRULABLE_KINDS = ("resolution", "measurement", "baseline") + MILESTONE_KINDS

#: Statuses that make a candidate DECIDED (Vocabulary). A candidate with no entry in one of these
#: (or with no entry at all) is undecided and is re-screened at its rank.
TERMINAL_STATUSES = ("decided", "final", "confirmed", "overruled", "skipped")

CANDIDATE_OPTIONS = ("ADOPT", "REJECT", "SKIP")
MILESTONE_OPTIONS = ("PASS", "WATCH", "FAIL")


# ------------------------------------------------------------------ the notification contract
def _field(text) -> str:
    """One notification field: no newline, no ``|`` (the separator), no leading/trailing space."""
    s = " ".join(str(text).split())
    return s.replace("|", "/")


def format_notify(tag: str, ident: str, fields) -> str:
    """``TAG <id> | field | field ...`` -- ONE physical line, absolute paths, no timestamp.

    The timestamp is prepended by ``Campaign.notify`` when the line is written to ``notify.log``
    (``%Y-%m-%d %H:%M ``), which is why it is not part of the format: a line this function produced
    and a line read back out of the log must both parse, and :func:`parse_notify` therefore accepts
    the stamped and unstamped spellings alike.

    ``tag`` must be one of :data:`NOTIFY_TAGS` -- a typo'd tag would be relayed as an unknown kind
    of event and quietly dropped by a filter somewhere downstream.
    """
    if tag not in NOTIFY_TAGS:
        raise ValueError(f"unknown notify tag {tag!r}; known: {NOTIFY_TAGS}")
    parts = [_field(f) for f in (fields or []) if str(f).strip()]
    head = f"{tag} {_field(ident)}" if str(ident).strip() else f"{tag}"
    return " | ".join([head] + parts)


def parse_notify(line: str) -> dict:
    """``{"stamp", "tag", "id", "fields": [...]}`` from a notification line.

    Accepts the line as :func:`format_notify` produced it and as ``notify.log`` stores it (with the
    ``%Y-%m-%d %H:%M`` stamp in front). ``stamp`` is ``None`` for an unstamped line. Raises on a
    line whose first token is not a known tag -- a parser that guessed would let a corrupted log
    line be relayed as a decision.
    """
    raw = str(line).strip()
    stamp = None
    parts = raw.split(" | ")
    head = parts[0].strip()
    tokens = head.split()
    if len(tokens) >= 3 and tokens[0][:1].isdigit() and ":" in tokens[1]:
        stamp = f"{tokens[0]} {tokens[1]}"
        tokens = tokens[2:]
    if not tokens or tokens[0] not in NOTIFY_TAGS:
        raise ValueError(f"not a notification line (unknown tag): {line!r}")
    tag = tokens[0]
    ident = " ".join(tokens[1:]).strip()
    return {"stamp": stamp, "tag": tag, "id": ident, "fields": [p.strip() for p in parts[1:]]}


# ------------------------------------------------------------------ ledger / state / override I/O
def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def read_ledger(cdir: Path) -> dict:
    """``{"schema": 1, "entries": [...]}`` -- the driver-owned decision record."""
    led = read_json(Path(cdir) / LEDGER_NAME, {})
    if not led:
        return {"schema": 1, "entries": []}
    led.setdefault("entries", [])
    return led


def write_ledger(cdir: Path, ledger: dict) -> None:
    """Atomically rewrite ``ledger.json`` (temp + ``os.replace``), driver-only.

    Serialised before anything is opened, so an unserialisable entry raises with the old file
    intact -- the same rule as ``Campaign.save``.
    """
    path = Path(cdir) / LEDGER_NAME
    payload = json.dumps(ledger, indent=2, sort_keys=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


def read_state(cdir: Path) -> dict:
    return read_json(Path(cdir) / STATE_NAME, {})


def append_override(cdir: Path, record: dict) -> str:
    """Append one override line, ``fsync``ed, with a fresh uuid; returns the uuid.

    Append-only + one JSON object per line + ``fsync`` is what makes a half-written line safe: the
    driver's ``ingest_overrides`` reads COMPLETE lines only and re-scans from byte 0 at every
    launch, deduping by the uuids it has already applied. A crash between the write and the flush
    therefore loses the answer (the user repeats it) instead of applying half of one.
    """
    rec = dict(record)
    rec.setdefault("uuid", str(_uuid.uuid4()))
    rec.setdefault("at", datetime.now().isoformat(timespec="seconds"))
    path = Path(cdir) / OVERRIDES_NAME
    # A TORN last line (a crash between the write and the fsync) ends without a newline. Appending
    # straight onto it would CONCATENATE the two, destroying this record as well as the fragment --
    # exactly when the user is repeating an answer BECAUSE the first one was lost. One leading
    # newline closes the fragment: `read_overrides` then skips it as unparseable and reads this
    # line whole.
    torn = False
    try:
        if path.stat().st_size:
            with path.open("rb") as probe:
                probe.seek(-1, os.SEEK_END)
                torn = probe.read(1) != b"\n"
    except OSError:
        torn = False
    with path.open("a") as fh:
        if torn:
            fh.write("\n")
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return rec["uuid"]


def read_overrides(cdir: Path) -> list[dict]:
    """Every COMPLETE override line, in order. A partial line is skipped, not guessed.

    The fragment need not be the LAST line: :func:`append_override` closes a torn one with a leading
    newline, so a repeated answer written after a crash sits on its own line below it.
    """
    path = Path(cdir) / OVERRIDES_NAME
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue          # a torn last line: the next launch re-reads the file from 0
        if isinstance(rec, dict) and rec.get("uuid"):
            out.append(rec)
    return out


# ------------------------------------------------------------------ small ledger queries (pure)
def entries(ledger: dict) -> list[dict]:
    return list(ledger.get("entries") or [])


def by_id(ledger: dict, ident: str) -> dict | None:
    for e in entries(ledger):
        if e.get("id") == ident:
            return e
    return None


def newest_for_candidate(ledger: dict, cid: str, *, skip_squashed: bool = True) -> dict | None:
    """The candidate's newest entry (its EFFECTIVE entry when ``skip_squashed``).

    "Newest" is write order, which is the ledger's own order -- ids are assigned in write order and
    a re-screen appends rather than replaces.
    """
    found = None
    for e in entries(ledger):
        if e.get("candidate") != cid:
            continue
        if skip_squashed and e.get("status") == "squashed":
            continue
        found = e
    return found


def effective_choice(entry: dict | None):
    """The choice that STANDS for an entry: the override's when one was applied, else ``chosen``."""
    if not entry:
        return None
    ov = entry.get("override") or {}
    if ov.get("choice") is not None:
        return ov["choice"]
    return entry.get("chosen")


def confirm_choice(entry: dict | None):
    """What ``confirm`` freezes -- which is NOT always the choice that currently stands.

    The plan's transition table has one asymmetric row: ``overruled(driver) -> confirmed`` means
    "user ``confirm`` freezes the entry's ``chosen``, i.e. REVERTS the driver's override +
    resurrection". ``overruled(driver)`` is ``override.by in {driver, milestone}`` (plan:545) -- a
    pooled self-overrule or a milestone drop. For those, ``confirm`` restores the entry's own pick;
    for everything else it freezes what stands.
    """
    if not entry:
        return None
    ov = entry.get("override") or {}
    if entry.get("status") == "overruled" and ov.get("by") in ("driver", "milestone"):
        return entry.get("chosen")
    return effective_choice(entry)


def unoverrulable_note(entry: dict | None) -> str | None:
    """Why ``confirm``/``overrule`` is refused for this entry, and what the verb for it IS.

    One message per reason: a measurement is not a choice at all, a milestone is a choice with a
    DIFFERENT verb. Shared by both CLI verbs and by the driver's ``_apply_decision_override``, so
    the refusal a user reads at the terminal and the one relayed in ``OVERRIDE-APPLIED`` agree.
    """
    kind = (entry or {}).get("kind")
    if kind in MILESTONE_KINDS:
        mid = (entry or {}).get("candidate") or "<Mid>"
        return (f"{(entry or {}).get('id')} is a {kind} entry: a milestone is not confirmed or "
                f"overruled. Its verb is `resume-milestone {mid} --accept` (accept the recipe as it "
                f"stands) or `resume-milestone {mid} --drop <Cid>` (drop an adoption in its block). "
                "Overruling it would re-book the promotion windows and then refuse the close.")
    if kind in UNOVERRULABLE_KINDS:
        return (f"{(entry or {}).get('id')} is a {kind} entry: measurements and baselines are not "
                "choices and cannot be confirmed or overruled. `show` lists the candidate/seed2 ids.")
    return None


def milestone_block(cands: dict, mid: str) -> list[str]:
    """The candidate ids whose decisions belong to milestone ``mid``'s block.

    The block is the rank range that ENDS at this milestone's ``after`` and starts after the
    previous milestone's ``after`` -- exactly the adoptions the milestone's batched pair confirms,
    and so exactly the set a ``--drop`` may name.
    """
    order = [c["id"] for c in cands.get("candidates") or []]
    miles = list(cands.get("milestones") or [])
    prev_after = None
    for mstone in miles:
        if mstone.get("id") == mid:
            after = mstone.get("after")
            lo = 0 if prev_after is None else order.index(prev_after) + 1
            hi = order.index(after) + 1 if after in order else len(order)
            return order[lo:hi]
        prev_after = mstone.get("after")
    raise KeyError(f"unknown milestone {mid!r} (known: {[m.get('id') for m in miles]})")


def unit_is_active(unit: str | None) -> bool:
    if not unit:
        return False
    try:
        out = subprocess.run(["systemctl", "--user", "is-active", str(unit)],
                             capture_output=True, text=True)
    except OSError:
        return False
    return out.stdout.strip() == "active"


def _applied_note(cdir: Path) -> str:
    st = read_state(cdir)
    unit = ((st.get("ladder") or {}).get("unit"))
    if unit_is_active(unit):
        return f"driver unit {unit} is running — applied within one poll (<= 120 s)"
    return "driver not running — applied at next launch"


# ------------------------------------------------------------------ verbs
class CandidatesMissing(SystemExit):
    """``candidates.yaml`` is not where the verb looked -- a message, not a traceback."""


def _load_cands(path: Path) -> dict:
    import yaml  # the harness's own dependency; imported here so `show`/`stop` need no yaml
    path = Path(path)
    if not path.exists():
        raise CandidatesMissing(
            f"{path} does not exist: `decide` and `resume-milestone` read the campaign's "
            "candidates.yaml to check the id they are given. Point --campaign-dir at the ladder "
            "directory, or pass --candidates <file>.")
    return yaml.safe_load(path.read_text())


def _eval_sha_field(record: dict) -> str:
    """``eval_sha`` of a ledger entry or campaign record, or :data:`PREDATES_SUITE` when it has none."""
    return str(record.get("eval_sha") or PREDATES_SUITE)


def _names(values) -> str:
    return ",".join(str(v) for v in values) if values else "none"


def _trained_field(entry: dict) -> str:
    """``trained:`` for ``show`` -- the categories the entry's comparison decided on (plan D4/D5).

    Read from the entry's evidence: ``decisive`` (the CANDIDATE's decisive categories) and
    ``untrained: {candidate, base}``, as rule v2 decisions record them; a baseline / rescreen entry
    that carries a screen reads its ``trained_categories`` (``decisive`` + ``watch`` with notes).
    ``—`` for an entry that predates the per-category suite (nothing was recorded).
    """
    ev = entry.get("evidence") or {}
    tc = ev.get("trained_categories") or (ev.get("screen") or {}).get("trained_categories") or {}
    decisive = ev.get("decisive")
    if decisive is None and isinstance(tc, dict):
        decisive = tc.get("decisive")
    if decisive is None:
        return "—"
    out = _names(decisive)
    watch = tc.get("watch") if isinstance(tc, dict) else None
    if watch:
        notes = watch.items() if isinstance(watch, dict) else ((c, None) for c in watch)
        out += " (watch: " + ", ".join(f"{c} {note}" if note else str(c) for c, note in notes) + ")"
    untrained = ev.get("untrained") or {}
    if isinstance(untrained, dict):
        parts = [f"{side} {_names(cats)}" for side, cats in untrained.items() if cats]
        if parts:
            out += " | untrained: " + "; ".join(parts)
    return out


def cmd_show(cdir: Path, args) -> int:
    led = read_ledger(cdir)
    st = read_state(cdir)
    lad = st.get("ladder") or {}
    rows = entries(led)
    if args.ident:
        rows = [e for e in rows if e.get("id") == args.ident or e.get("candidate") == args.ident]
        if not rows:
            print(f"no ledger entry with id or candidate {args.ident!r}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"campaign {cdir}  phase {st.get('phase')}  unit {lad.get('unit')}  code_sha {lad.get('code_sha')}  "
          f"eval_sha {_eval_sha_field(lad)}")
    print(f"pending pairs: {lad.get('pending') or []}   user_choice: "
          f"{ {k: v.get('choice') for k, v in (lad.get('user_choice') or {}).items()} }")
    print("| id | kind | candidate | chosen | eff | status | conf | pooled | experiment | frozen |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for e in rows:
        print(f"| {e.get('id')} | {e.get('kind')} | {e.get('candidate') or ''} | {e.get('chosen')} | "
              f"{effective_choice(e)} | {e.get('status')}"
              f"{'/' + e['sub_status'] if e.get('sub_status') else ''} | {e.get('confidence') or ''} | "
              f"{e.get('pooled_confidence') or ''} | {e.get('experiment') or ''} | "
              f"{'yes' if e.get('frozen') else ''} |")
    for e in rows:
        # The eval that produced the entry's evidence (D8) and what its comparison decided on (D5).
        print(f"- {e.get('id')}: eval_sha: {_eval_sha_field(e)} | trained: {_trained_field(e)}")
    for e in rows:
        # A milestone is NOT advertised here: its verb is `resume-milestone`, and advertising
        # confirm/overrule beside it is what made `overrule M1 --choose FAIL` look like the answer.
        if e.get("kind") in MILESTONE_KINDS:
            mid = e.get("candidate") or "<Mid>"
            print(f"- {e.get('id')}: a milestone — `resume-milestone {mid} --accept` (only while it "
                  f"is `await`) or `resume-milestone {mid} --drop <Cid>`; never confirm/overrule")
            continue
        if e.get("kind") in UNOVERRULABLE_KINDS:
            continue
        print(f"- {e.get('id')}: confirm/overrule verbs available (options {e.get('options')})")
    return 0


def cmd_pending(cdir: Path, args) -> int:
    led = read_ledger(cdir)
    st = read_state(cdir)
    lad = st.get("ladder") or {}
    print(f"phase {st.get('phase')}   unit {lad.get('unit')} ({'active' if unit_is_active(lad.get('unit')) else 'inactive'})")
    cur = st.get("current") or {}
    if cur:
        print(f"current: {cur.get('tag')} ({cur.get('kind')}) phase {cur.get('phase')} iters {cur.get('iters')}")
    pend = lad.get("pending") or []
    print(f"pending pairs ({len(pend)}): {pend}")
    for did in pend:
        e = by_id(led, did) or {}
        print(f"  {did}: {e.get('candidate')} chosen {effective_choice(e)} ({e.get('status')}) — "
              f"confirm {did} / overrule {did} --choose <other>")
    awaits = [e for e in entries(led) if e.get("status") == "await"]
    for e in awaits:
        print(f"awaiting user: {e.get('id')} ({e.get('candidate')}) sub {e.get('sub_status')} — "
              f"resume-milestone {e.get('candidate')} --accept | --drop <Cid>")
    if str(st.get("phase") or "").startswith("await_user"):
        print(f"driver is parked: {st.get('phase')} (see {cdir}/notify.log)")
    if (Path(cdir) / STOP_NAME).exists():
        print(f"STOP file present ({cdir}/{STOP_NAME}) — the driver stops at its next check; `unstop` removes it")
    return 0


def _refuse_skipped(entry: dict, verb: str) -> str | None:
    """A SKIP entry is not a choice a user may freeze or flip -- the verb for it is ``decide run``.

    The driver's transition table has no ``skipped -> confirmed`` / ``skipped -> overruled`` row
    (only ``skipped -> squashed``), so without this the CLI would accept the line, print
    "(frozen)", and the driver would refuse it one poll later with an illegal-transition note.
    """
    if entry.get("status") != "skipped":
        return None
    cid = entry.get("candidate") or "<Cid>"
    return (f"{entry.get('id')} is a SKIP entry ({cid} was not screened), not a decision: `{verb}` "
            f"has nothing to freeze or flip. To run it anyway use `decide {cid} --choose run`.")


def _refuse_squashed(entry: dict, verb: str) -> str | None:
    """A SQUASHED entry is a VOIDED record, not a standing choice -- the verb for it is ``decide run``.

    The driver's transition table lets only ``derived`` move a ``squashed`` entry (that move is the
    resurrection), so without this the CLI accepted the line with rc 0 and printed "(frozen)", and
    the driver refused it one poll later inside an ``OVERRIDE-APPLIED`` note -- the same split
    :func:`_refuse_skipped` closes for SKIP entries, and a worse one to read: the entry the user
    was answering had already been voided by a recipe move, so freezing it could not have been the
    answer they wanted either way.
    """
    if entry.get("status") != "squashed":
        return None
    cid = entry.get("candidate") or "<Cid>"
    why = next((h.get("why") for h in reversed(entry.get("history") or [])
                if h.get("to") == "squashed"), None)
    return (f"{entry.get('id')} is SQUASHED ({why or 'the recipe it was decided on has moved'}): a "
            f"voided record is not a choice, so `{verb}` has nothing to freeze or flip. To screen "
            f"{cid} again use `decide {cid} --choose run`; `show` lists the live entries.")


def cmd_confirm(cdir: Path, args) -> int:
    led = read_ledger(cdir)
    e = by_id(led, args.ident)
    if e is None:
        print(f"no ledger entry {args.ident!r} (see `show`)", file=sys.stderr)
        return 2
    refusal = unoverrulable_note(e)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    refusal = _refuse_skipped(e, "confirm") or _refuse_squashed(e, "confirm")
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    # PLAN_AMENDMENTS 24: the plan's table has `overruled(driver) -> confirmed` and no
    # `overruled(user) -> confirmed`. On a driver/milestone override `confirm` REVERTS it; on the
    # user's own overrule there is nothing to revert, and it would freeze the entry's pre-override
    # pick instead of the answer the user gave.
    if e.get("status") == "overruled" and (e.get("override") or {}).get("by") == "user":
        print(f"{args.ident} is already overruled BY YOU and frozen at "
              f"{effective_choice(e)!r}: `confirm` is the revert of a driver or milestone override "
              f"and has nothing to revert here. To pick again use "
              f"`overrule {args.ident} --choose <choice> --why ...`.", file=sys.stderr)
        return 2
    choice = confirm_choice(e)
    uid = append_override(cdir, {"decision": args.ident, "action": "confirm",
                                 "choice": choice, "why": args.why or "user confirm"})
    print(f"confirm {args.ident} -> {choice} (frozen); uuid {uid}")
    if choice != effective_choice(e):
        print(f"note: {args.ident} stands at {effective_choice(e)} under a "
              f"{(e.get('override') or {}).get('by')} override; `confirm` REVERTS it to the entry's "
              f"own pick {choice} and resurrects what that override squashed (plan transition "
              f"table). To freeze {effective_choice(e)} instead use "
              f"`overrule {args.ident} --choose {effective_choice(e)} --why ...`.")
    print(_applied_note(cdir))
    return 0


def cmd_overrule(cdir: Path, args) -> int:
    led = read_ledger(cdir)
    e = by_id(led, args.ident)
    if e is None:
        print(f"no ledger entry {args.ident!r} (see `show`)", file=sys.stderr)
        return 2
    refusal = unoverrulable_note(e)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    refusal = _refuse_skipped(e, "overrule") or _refuse_squashed(e, "overrule")
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    options = e.get("options") or list(CANDIDATE_OPTIONS)
    if args.choose not in options:
        print(f"--choose {args.choose!r} is not one of {args.ident}'s options {options}", file=sys.stderr)
        return 2
    if not args.why:
        print("overrule needs --why (it is the record of a branch mispredict)", file=sys.stderr)
        return 2
    uid = append_override(cdir, {"decision": args.ident, "action": "overrule",
                                 "choice": args.choose, "why": args.why})
    print(f"overrule {args.ident} -> {args.choose}; uuid {uid}")
    print(_applied_note(cdir))
    return 0


def cmd_decide(cdir: Path, args) -> int:
    cands = _load_cands(args.candidates or (Path(cdir) / "candidates.yaml"))
    known = [c["id"] for c in cands.get("candidates") or []]
    if args.ident not in known:
        print(f"{args.ident!r} is not a candidate id (known: {known})", file=sys.stderr)
        return 2
    if args.choose not in ("run", "skip"):
        print("decide --choose takes run or skip", file=sys.stderr)
        return 2
    led = read_ledger(cdir)
    newest = newest_for_candidate(led, args.ident)
    if args.choose == "skip" and newest is not None and newest.get("kind") in ("candidate", "seed2") \
            and effective_choice(newest) in ("ADOPT", "REJECT"):
        print(f"{args.ident} already carries a DECISION ({newest.get('id')} -> "
              f"{effective_choice(newest)}, {newest.get('status')}): `decide --choose skip` is for an "
              f"undecided candidate. Use `confirm {newest.get('id')}` to freeze it or "
              f"`overrule {newest.get('id')} --choose REJECT --why ...` to change it.", file=sys.stderr)
        return 2
    uid = append_override(cdir, {"candidate": args.ident, "choice": args.choose,
                                 "why": args.why or f"user decide {args.choose}"})
    print(f"decide {args.ident} -> {args.choose}; uuid {uid}")
    if args.choose == "run" and newest is not None:
        print(f"note: {newest.get('id')} ({effective_choice(newest)}) will be squashed on ingest and "
              f"{args.ident} re-screened at its rank")
    print(_applied_note(cdir))
    return 0


def cmd_resume_milestone(cdir: Path, args) -> int:
    cands = _load_cands(args.candidates or (Path(cdir) / "candidates.yaml"))
    try:
        block = milestone_block(cands, args.ident)
    except KeyError as exc:
        print(exc.args[0] if exc.args else str(exc), file=sys.stderr)
        return 2
    if bool(args.accept) == bool(args.drop):
        print("resume-milestone takes exactly one of --accept or --drop <Cid>", file=sys.stderr)
        return 2
    led = read_ledger(cdir)
    if args.drop:
        if args.drop not in block:
            print(f"--drop {args.drop} is not in {args.ident}'s block {block}", file=sys.stderr)
            return 2
        e = newest_for_candidate(led, args.drop)
        if e is None or effective_choice(e) != "ADOPT":
            print(f"--drop is refused unless {args.drop} is an ADOPT in {args.ident}'s block "
                  f"(it is {effective_choice(e) if e else 'undecided'}): dropping a REJECT or a SKIP "
                  "would not change the recipe and the milestone would re-run unchanged.", file=sys.stderr)
            return 2
        uid = append_override(cdir, {"milestone": args.ident, "drop": args.drop,
                                     "why": args.why or f"{args.ident} drop {args.drop}"})
        print(f"resume-milestone {args.ident} --drop {args.drop}; uuid {uid}")
    else:
        # `--accept` FREEZES every unfrozen adoption in the block (plan:552-556 defines the row only
        # FROM `await`), and a frozen ADOPT can no longer be dropped by a milestone contradiction
        # nor moved by a pooled self-overrule. Typed early -- or re-run from an OLD notification
        # after the milestone closed -- it silently disarms the campaign's own error correction, so
        # it is refused unless the milestone is actually waiting for this answer.
        e = newest_for_candidate(led, args.ident)
        # Already accepted: the plan's row is `await -> accepted_i`, ONCE. The status stays `await`
        # while the promotion is queued, so a second `--accept` (a re-typed answer, or an old
        # notification run again) passes the guard below and re-applies -- freezing nothing new but
        # adding a history row and reporting `frozen adoptions []`. rc 0 with the note: the user's
        # answer stands, there is simply nothing left to append.
        if e is not None and e.get("status") == "await" and e.get("sub_status") == "accepted_i":
            print(f"{args.ident} is already accepted (every unfrozen adoption in {block} was frozen "
                  f"then) and its promotion is queued: `--accept` is answered once. `pending` shows "
                  f"what is actually waiting; to change the recipe use "
                  f"`resume-milestone {args.ident} --drop <Cid>`.")
            print(_applied_note(cdir))
            return 0
        if e is None or e.get("status") != "await":
            print(f"{args.ident} is not awaiting an answer (it is "
                  f"{'not in the ledger yet' if e is None else repr(e.get('status'))}): `--accept` "
                  f"freezes every unfrozen adoption in {block} as confirmed and is legal only on a "
                  f"(i) or (ii) FAIL, which the MILESTONE notification quotes. `pending` lists what "
                  f"is actually waiting.", file=sys.stderr)
            return 2
        uid = append_override(cdir, {"milestone": args.ident, "accept": True,
                                     "why": args.why or f"{args.ident} accept"})
        print(f"resume-milestone {args.ident} --accept (every unfrozen adoption in {block} is frozen "
              f"as confirmed); uuid {uid}")
    print(_applied_note(cdir))
    return 0


def cmd_stop(cdir: Path, args) -> int:
    path = Path(cdir) / STOP_NAME
    path.write_text(f"{datetime.now().isoformat(timespec='seconds')} {args.why or 'ledger_cli stop'}\n")
    print(f"wrote {path}: the driver stops at its next check (a running child is SIGTERMed, then "
          f"SIGKILLed after 120 s) and parks at phase=await_user:stop")
    print(_applied_note(cdir))
    return 0


def cmd_unstop(cdir: Path, args) -> int:
    path = Path(cdir) / STOP_NAME
    if path.exists():
        path.unlink()
        print(f"removed {path}")
    else:
        print(f"no {path}")
    # Like every other verb: whether the driver sees this now or at the next launch. `unstop` is
    # the one a user types expecting the campaign to RESUME, and a parked driver does not.
    print(_applied_note(cdir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--candidates", default=None, help="default: <campaign-dir>/candidates.yaml")
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("show"); p.add_argument("ident", nargs="?"); p.add_argument("--json", action="store_true")
    sub.add_parser("pending")
    p = sub.add_parser("confirm"); p.add_argument("ident"); p.add_argument("--why", default=None)
    p = sub.add_parser("overrule"); p.add_argument("ident"); p.add_argument("--choose", required=True)
    p.add_argument("--why", default=None)
    p = sub.add_parser("decide"); p.add_argument("ident"); p.add_argument("--choose", required=True)
    p.add_argument("--why", default=None)
    p = sub.add_parser("resume-milestone"); p.add_argument("ident")
    p.add_argument("--accept", action="store_true"); p.add_argument("--drop", default=None)
    p.add_argument("--why", default=None)
    p = sub.add_parser("stop"); p.add_argument("--why", default=None)
    sub.add_parser("unstop")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cdir = Path(args.campaign_dir)
    if not cdir.exists():
        print(f"no campaign directory {cdir}", file=sys.stderr)
        return 2
    verbs = {"show": cmd_show, "pending": cmd_pending, "confirm": cmd_confirm, "overrule": cmd_overrule,
             "decide": cmd_decide, "resume-milestone": cmd_resume_milestone, "stop": cmd_stop,
             "unstop": cmd_unstop}
    try:
        return verbs[args.verb](cdir, args)
    except CandidatesMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
