#!/usr/bin/env python3
"""Write ``body_cmd_train/candidates.yaml`` -- the body chain as run_candidates REFERENCE chains (PLAN steps 5, 6, 8).

Every stage overlay is the SHIPPED lineage's overlay, verbatim (ppo_construction FB6k__B / SF__B_c4_s18 /
SF__B_rest_s18 stages, phase-2 P2__S1_s18 p2a), with exactly two edits: ``KRABBY_OMNI_FACING`` dropped (refused in
body mode: the facing band has no meaning without a facing target) and the frozen envelope knobs added
(``envelope_freeze.json``). The bases are the ``_b`` presets of the lineage's bases (1b -> 1b_b, 1a -> 1a_b,
2a -> 2a_b). ONE reference per stage per seed, each resuming the previous stage's reference head, so the
orchestrator reads every stage end (PLAN step 6) before the next starts -- the lineage ran c5 -> c6 -> c6x as one
chain with ``previous_stage`` resumes; a reference resume is the same checkpoint resume (the PPO-counter fix on this
branch carries priv_reg across both).

Stage evals: ``suite_body`` (suite__omni_body) at every stage end -- including c5, which the lineage did not evaluate
(an eval only, nothing trained differs). Smoke chains (PLAN step 5): 100 iterations from scratch on 1b_b + FB6k,
1a_b + c4, 2a_b + p2a (the cheap p2a_b smoke the audit deferred), no driver eval.

    bc_make_candidates.py [--envelope envelope_freeze.json]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
C = HERE.parent
EXP = C.parent
SEEDS = (18, 16, 17)
BASE_B = {"1b": "1b_b", "1a": "1a_b", "2a": "2a_b"}
DROP = ("KRABBY_OMNI_FACING",)


def shipped_stages() -> dict:
    pc = yaml.safe_load((EXP / "2026-10-01_ppo_construction/candidates.yaml").read_text())["chains"]
    p2 = yaml.safe_load((EXP / "2026-10-07_omni_phase2/candidates.yaml").read_text())["chains"]
    st = {"fb6k": pc["FB6k__B"]["stages"][0], "c4": pc["SF__B_c4_s18"]["stages"][0]}
    for s in pc["SF__B_rest_s18"]["stages"]:
        st[s["stage"]] = s
    st["p2a"] = p2["P2__S1_s18"]["stages"][0]
    return st


def body_overlay(ov: dict, knobs: dict) -> dict:
    out = {k: v for k, v in ov.items() if k not in DROP}
    out.update(knobs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelope", default=str(C / "envelope_freeze.json"))
    a = ap.parse_args()
    knobs = json.loads(Path(a.envelope).read_text())["knobs"]
    st = shipped_stages()
    order = ("fb6k", "c4", "c5", "c6", "c6x", "p2a")
    chains, refs = {}, {}

    def stage(name: str, iters: int, resume, evals, from_scratch=False) -> dict:
        s = st[name]
        return {"stage": name, "base": BASE_B[s["base"]], "iters": iters, "from_scratch": from_scratch,
                "resume_from": resume, "overlay": body_overlay(s["overlay"], knobs), "eval": list(evals),
                "categories": list(s["categories"]), "floor": {"kind": "none"}}

    for seed in SEEDS:
        prev = None
        for name in order:
            cid = f"BC_{name}_s{seed}"
            s = st[name]
            resume = "scratch" if prev is None else {"reference": prev}
            chains[cid] = {"notes": f"body-cmd {name}_b (PLAN step {'6' if seed == 18 else '8'}): the shipped "
                                    f"{name} stage overlay minus KRABBY_OMNI_FACING plus the frozen envelope, base "
                                    f"{BASE_B[s['base']]}.",
                           **({"absolute_units": True} if name == "p2a" else {}),
                           "stages": [stage(name, int(s["iters"]), resume, ["suite_body"],
                                            from_scratch=prev is None)]}
            refs[cid] = {"chain": cid, "seed": seed}
            prev = cid
    # c6_b-stop diagnostic (CHARTER 2026-10-16): two 500-it forks of the c6_b head on the c6 overlay -- control, and
    # the instantaneous tent (KRABBY_OMNI_BODY_TENT_RATE=instant). Checkpoint forks (no reference / code_sha lookup).
    c6b = C / "stage_reads" / "BC_c6_s18.json"
    if c6b.is_file():
        ck = json.loads(c6b.read_text())["ckpt"]
        ck = str(Path(ck).relative_to("/home/nickmagus/krabby/krabby-research"))
        for sid, extra in (("DX_ctrl", {}), ("DX_inst", {"KRABBY_OMNI_BODY_TENT_RATE": "instant"})):
            cid = f"BC_{sid}_s18"
            stg = stage("c6", 500, {"checkpoint": ck}, ["suite_body"])
            stg["overlay"].update(extra)
            chains[cid] = {"notes": f"body-cmd c6_b-stop diagnostic: 500 its from the c6_b head, c6_b overlay"
                                    f"{' + instant tent' if extra else ' (control)'}.", "stages": [stg]}
            refs[cid] = {"chain": cid, "seed": 18}
    for sid, name in (("SMK_1b", "fb6k"), ("SMK_c4", "c4"), ("SMK_p2a", "p2a")):
        cid = f"BC_{sid}_s18"
        chains[cid] = {"notes": f"body-cmd smoke (PLAN step 5): 100 its from scratch, {name}_b overlay.",
                       "stages": [stage(name, 100, "scratch", [], from_scratch=True)]}
        refs[cid] = {"chain": cid, "seed": 18}
    doc = {
        "schema": 1,
        "plant": "A15+B",
        "seeds": {"screen": 18, "confirm": 16},
        "screen": {"chain": "BC_SMK_1b_s18", "eval": [], "iters": 100},
        "promotion": {"enabled": False},
        "baseline": {"id": "BC_SMK_1b_s18", "chain": "BC_SMK_1b_s18"},
        # phase 2's decision rule verbatim (the driver needs a complete rule_version-2 block to validate); only its
        # text changes -- it is RECORD ONLY here, as it was there.
        "decision_rule": {**copy.deepcopy(yaml.safe_load((EXP / "2026-10-07_omni_phase2/candidates.yaml").read_text())
                                          ["decision_rule"]),
                          "text": "BODY-COMMAND CAMPAIGN (2026-10-15_body_cmd): RECORD ONLY. The campaign's real "
                                  "gates are PLAN.md's pre-registered gates 0-7 and the step-6 stage-end rules, "
                                  "applied by scripts/ readers against the anchors in the CHARTER; this driver "
                                  "only trains the reference chains and runs their stage evals."},
        "chains": chains,
        "references": refs,
        "candidates": [],
        "milestones": [],
    }
    out = C / "body_cmd_train" / "candidates.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# GENERATED by scripts/bc_make_candidates.py -- do not edit by hand.\n"
                   + yaml.safe_dump(doc, sort_keys=False, width=120))
    print(f"{out}: {len(chains)} chains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
