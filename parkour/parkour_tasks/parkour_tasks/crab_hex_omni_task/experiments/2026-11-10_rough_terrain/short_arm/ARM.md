# Amendment A: short arm (same-mix short fine-tunes, select on es1–2, confirm on es5–8)

> **Status:** audited, 2 passes (cap). Pass 1: 4 blocking, all fixed. Pass 2: 1 blocking (the G FAIL branch skipped step 10), fixed; that fix was not re-audited.
> **Routing:** opus session, roster ok. Fable checks run at steps 6 and 10 under the standing exception.
> **Honest odds:** ACCEPT is about 8 %. The most likely good outcome is G PASS with R failing on yaw or back (HALT_R); c1 sits on both lines.
> - Every early RT_s19 checkpoint sits on the turn yaw_progress R line: worse by 0.0069 / 0.0065 / 0.0081 against 0.0066.
> - That line is not seed-stable. Its per-seed noise is its se_diff, ≈ 0.0007–0.0009, so a candidate on the line passes it on both fresh seeds only about a quarter of the time (c1: ≈ 0.24).
> - What the arm decides cleanly is whether the two early terrain gains (slopes, ledge_down) replicate on fresh seeds and on a second training seed.

## Context

**Step 9.** RT_s19 `model_34990` was NOT ACCEPTED (`results/step9/gates.json`).
- G: 1 of 5 kinds gains, and ledge_down regresses.
- R: back track_err (R 0.0617) and turn yaw_progress (R 0.0066) fail.

**Sweep** (`results/sweep/sweep.json`, report only).
- At 29500 (+509 its): slopes gains +0.190 (lower bound 0.125) and ledge_down reads +0.185 (SE 0.035). But yaw_progress is worse by 0.0069 against R 0.0066 on es1.
- At 31000: inside every es1 line, but ledge_down is only +0.055.
- These were picked among 4 checkpoints on the same episodes, so they are a selection, not a result.

**User decision (2026-11-11): "Plan short arm".** Fine-tune TDR about 500–1000 its on the same mix, with the gates fixed, then evaluate both teachers on new test seeds.

**Facts the design rests on.**
- Same-seed training replays (memory note), so a seed-19 run on `RT_train.env` would reproduce RT_s19's existing checkpoints. The runner uses max_iterations only as its loop bound (`on_policy_runner_with_extractor.py:312`).
- TDR's fresh-seed side is already running and unread: block `tdrfresh` (scratch `rt_tdr_fresh.py`), 33 rows `RTF_03c3e472_*` at es+4, started 09:51.

## Settled

1. **Gates frozen.**
   - `thresholds_rough_v1.json` stays byte-unchanged (sha ce45e01b), as do `readout.py`, `rt_spec.py` and `make_thresholds.py`.
   - G and R constants, k = 2 and the PLAN decision table all apply.
2. **Same terrain mix.** `envs/RT_train.env` stays byte-unchanged (26a0461e); no knob changes.
3. **Build unchanged.** No file under `parkour/` outside `experiments/` changes, so code_sha 89c1916b and eval_sha 9fe964d7 hold. Never push; no bake without a yes; decide-record-notify; one queue; memory caps.

## Design

**Pool:** four candidates, two seeds × two lengths, pre-registered.

| id | checkpoint | its after TDR | evidence |
|---|---|---|---|
| c1 | RT_s19 `model_29500` (6d1f29be) | +509 | screening rows exist (RTSW) |
| c2 | RT_s19 `model_30000` (a346185c) | +1009 | never evaluated |
| c3 | RT_s20s `model_29500` | +509 | new run |
| c4 | RT_s20s `model_30000` | +1009 | new run (final) |

**New run RT_s20s.**
- Resumes TDR (`model_28991`, 03c3e472, counter 30450) on `RT_train.env`: seed 20, 1010 its (28991 → 30000), 256 envs, ≈ 0.7 h.
- `short_arm/rt_short_train.py` sets `M.TRAIN_NAME`, `TRAIN_ITERS` and `TRAIN_SEED` to RT_s20s, 1010 and 20, then calls `rt_train.main`.
  - The pre-registration guard (`rt_train.py:141`) and PF14 (seed, max_iterations, run_name) then enforce the arm's values.
  - `rt_train.py` itself is not edited.
- The iteration-1500 line does not fire, which is expected.

**Screening** (selection seeds, already spent).
- Rows per candidate: slopes b2 and ledge_down b2 at es1–2, plus `R_suite` es1–2 (user, 2026-11-11: "Approve + es2 suite"). They are paired with TDR's RT9 and RT6 stamps.
- c1 reuses its RTSW rows and adds only its es2 suite row, so 19 new rows.

**Pick rule** (mechanical).
- G lines come from `readout.g_line` with the frozen G rule. Suite lines come from `gate_read.zero_kg_check` on es1 and es2, drift 0.
- z(c) = the minimum, over the 22 decisive suite lines (11 per seed), of (margin − worse_by) / se_diff, with each line's own se_diff from the same call. So the slack is in noise units for every line, not as a fraction of margin.
- Q = slopes gains and ledge_down gains, and neither regresses. E = Q ∩ {z > 0}.
- The pick:
  1. If any E member has z ≥ 1, take the largest ledge_down diff among those members.
  2. Otherwise, if E is not empty, take the largest z.
  3. Otherwise (Q only), take the largest z in Q, and record "R tracking predicted marginal" before the confirm.
- Ties: a ledge_down diff within 0.01 in tier 1, or a z within 0.1 in tiers 2–3, goes to fewer iterations, then to seed 19.
- On es1, c1 ∈ Q \ E. Its yaw worse_by is 0.006887 against R 0.006606, with se_diff ≈ 0.0007, so z ≈ −0.4. Q is not empty, so a pick always exists.

**Confirm** (decisive). The pick against TDR's RTF rows, G first:
- **confirm-1** (12 rows): the 5 gate bands and `R_suite`, at es5–6. G is read here.
- **confirm-2** (21 rows): hills b3 and gravel b3 at es5–6; field_hard at es5–8; field_light, ring_obst, field_turn, L090, L135 and ×0.7 L090 at es5–6; side22 at es5.

**Readout:** `short_arm/readout_fresh.py`.
- It subclasses `readout.Rows` and overrides only `found()`, with es → {1:5, 2:6, 3:7, 4:8}[es], then calls `readout.gates()` unchanged.
- That one point covers every seed lookup: gate_env_seeds, rows_rt env_seeds, zero_kg_inputs' suite (1, 2) and side22 (1), and `ZERO_KG_FIELD`.
- It fails closed when:
  - any real es falls outside 5–8;
  - a stamp's build differs from the pin;
  - the thresholds sha differs from ce45e01b;
  - a frozen-script sha has changed;
  - any input to the R tracking lines has changed since c2054a8c: `experiments/tools/pipeline/` and the Task-4 `thresholds_v1_2.json` (969b2988), `threshold_inputs.yaml` and `pipeline.yaml`. `select.py` applies the same check;
  - selection.json is not yet committed (so RTF stays unread until then).
- A missing row reads INCOMPLETE.
- The 6000-it drift re-read is labelled not applicable (report only).

## Steps

0. **Approve the arm.** **[you]**
1. **Pre-register.** Write `short_arm/ARM.md` and a CHARTER entry with the pool, rule, seed map, rows, decision rules and predicted odds. State that this is the campaign's second decisive read. Commit locally before any c2–c4 eval number exists. **[main (opus) · medium — record text, no design left]**
2. **Build `short_arm/`.**
   - `rt_short_train.py`, plus `rt_short.py` with blocks `strain`, `ssel`, `sconf1` and `sconf2`, in the `rt_sweep.py` / `rt_tdr_fresh.py` pattern. **[sonnet · high — mechanical from this spec]**
   - `readout_fresh.py` and `select.py`. **[opus · high — the decisive reader; a slipped seed is silent]**
3. **CPU checks.** Any miss returns to step 2. **[sonnet · high — runs the listed checks]**
   - build check;
   - `make_thresholds.py --check`;
   - `readout_fresh` under the identity map reproduces `results/step9/gates.json` to 1e-12;
   - fail-closed tests: a missing stamp reads INCOMPLETE; an es1 stamp under the map is refused; a wrong thresholds sha is refused;
   - `select.py` reproduces the sweep's 29500 row and step 9's es1 suite lines;
   - dry passes give 1 / 19 / 12 / 21 slots with no collisions;
   - the trainer dry-run passes preflight with expected final `model_30000` and argv `--seed 20 --max_iterations 1010`;
   - PYTHONPATH in `/proc/<pid>/environ`.
4. **GPU: train RT_s20s** (≈ 0.7 h), through the queue and watcher. **[main]**
5. **GPU: selection rows** (19 rows, ≈ 1.7 h). Audit OK. **[main]**
6. **Select.** Run `select.py` and write `selection.json`. The fable recompute must agree before `selection.json` is committed and confirm-1 launches; on a disagreement, stop and ask. Then add a CHARTER line and notify. **[main (opus) · medium — mechanical rule; ✓ fable · high recompute, which gates the commit]**
7. **GPU: confirm-1** (≈ 0.9 h). **[main]**
8. **G read** via `readout_fresh`.
   - On G FAIL: NOT ACCEPTED, and confirm-2 does not run. Step 10 runs on the confirm-1 rows: G lines, report lines, the hash-check line and the fable re-derivation; lines that need confirm-2 read n/a. Then the CPU diagnosis, step 11 and HALT.
   - On G PASS: launch confirm-2 at once and notify the suite lines. **[opus · high]**
9. **GPU: confirm-2** (≈ 1.4 h). **[main]**
10. **Verdict** with report lines:
    - success beside completion;
    - the 0.14 m split;
    - slopes up / down;
    - back ratio, yaw rate and pitch;
    - selection → confirm shrinkage;
    - the seed × length table;
    - step 9's hash-check line: thresholds, frozen scripts, the R-tracking inputs, the build of every stamp, and each stamp's sha keyed to its checkpoint.
    **[opus · high; ✓ fable · high independent re-derivation from the raw stamps]**
11. **Record.** CHARTER entry, local commit, notify, HALT. **[sonnet · medium]**

## Gates (unchanged; the seeds move)

| Gate | Rows (n) | Pass line |
|---|---|---|
| G | 5 gate bands, es5–6 (200 per kind) | as PLAN; k = 2 |
| R completion / solved | field_hard es5–8 (400); the other 6 completion rows at es5–6; hills b3 and gravel b3 at es5–6 | new ≥ TDR − max(0.05, 2·SE) |
| R tracking | suite, 0 kg, es5–6 | v1.2 decisive set, drift 0 |
| R side | side22, es5 | FAST/FAST |
| Selection (not decisive) | slopes b2 and ledge_down b2 at es1–2; suite es1–2 | pick rule |

## Decision rules

| Outcome | Action |
|---|---|
| Selection | Decide, record, notify; no stop |
| confirm-1 G FAIL | NOT ACCEPTED; confirm-2 does not run; step 10 on the confirm-1 rows (as step 8); CPU diagnosis; HALT |
| confirm-1 G PASS | confirm-2 at once |
| G ∧ R | Accept; HALT for a yes; propose P2 (re-distil, re-pair, bake) |
| G passes, R fails | HALT_R; the user decides. Remedy by failing line: for yaw / move_turn, the env-only retention arm (flat + field ≈ 0.4, ORIENT_ALL_TILES after the replay gate, turn dose), read on es9+ with fresh TDR rows (RTF is not reused for a decisive read); for back, priced overshoot (a code change; the pin moves) |
| INVALID / INCOMPLETE | Re-run the failed slots at the same seeds |

**Never:**
- re-pick after any es5+ number exists;
- read a runner-up on es5–8;
- extend seeds;
- pool es1–2 into the read;
- gate on report lines.

Any later candidate needs es9+ and a new pre-registration.

**Before approval** nothing in steps 1–11 runs. The only work already running is RTF, TDR's fresh-seed rows, which stay unread until `selection.json` is committed.

**Odds:**
- ACCEPT ≈ 8 % (range 3–12 %);
- G pass ≈ 0.45;
- HALT_R ≈ 0.35;
- NOT ACCEPTED ≈ 0.55.

## Budget (wall-clock)

| Item | h |
|---|---|
| Steps 1–3 (CPU, overlapping RTF; RTF ends ≈ 12:10–12:30) | 1–1.5 |
| Training | 0.7 |
| Selection rows | 1.7 |
| confirm-1 | 0.9 |
| confirm-2 (G PASS only) | 1.4 |
| Reads, fable checks, records | 0.5 |
| **After RTF: G FAIL / G PASS** | **≈ 3.8 / ≈ 5.2 (GPU 3.3 / 4.7)** |

- Disk: 48 rows × 0.4 GB ≈ 19 GB, plus RTF's ≈ 13 GB, against 86 GB free.
- DerivedDataCache: 37 GB against an 80 GB cap.

## Left for implementation (audit pass 1, deferred; check each)

- **Seed labels.** `readout.gates()` labels stay logical (`suite es1`) while the real seeds are 5–6. Every output and the verdict must name the real seed.
- **Scan before unblinding.** `Rows.__init__` scans every stamp's metadata, RTF included. Construct `Rows` only after the selection.json-committed guard, or on a root that excludes `RTF_*`.
- **Seed guards.** The `found()` override raises on a logical es outside 1–4, and `readout_fresh` asserts the real es is 5–8. `find()` accepts a stamp with no `tp_row.env_seed`.
- **Import, don't copy.** `readout_fresh` imports `readout` from `scripts/`; the self-check hashes `HERE/readout.py`.
- **Suite-only reads** (step 8, `select.py`): take the per-line worse_by and margin for rows starting `suite` with `not watch`. Never use `res["verdict"]` or `res["fails"]`, which carry field and side22 fails while those rows are missing.
- **Checkpoint identity.** `rt_short.py` needs its own check for c1–c4; it must not reuse `plan_gate` / `G.ident`, which pin RT_s19 at 34990.
- **Launcher.** It imports `rt_machinery` under that exact name, passes `--name RT_s20s`, and runs through the queue or `--wait-idle` while RTF runs. Defaults bound at import time are seed 19.
- **le_noise clause.** Record in ARM.md that `readout.g_line`'s `and not le_noise` clause cannot bind (gain ≥ 0.10, regress < −0.05).
- **Counter restore** on resume relies on eabe7324, an ancestor of HEAD. Note it in ARM.md.
- **Iteration labels.** +509 / +1009 are offsets; the runs have 510 / 1010 completed updates. c1/c3 and c2/c4 are at parity.
- **Disk and DDC.** Re-read both before launch: 82 GB free and DDC 39 GB at audit time. Rows average ≈ 0.23 GB.
- **Status line scope.** Only the pick is read on fresh seeds. Seed 20's evidence is its es1–2 screen. The odds are rough: ACCEPT + HALT_R ≈ G pass.

- **Pass 2 deferred** (check each):
  - z is +inf when se_diff = 0 and worse_by ≤ margin; the survival lines hit this.
  - `zero_kg_check` keys stay logical (`suite es1`), because v1.2's drift dict is keyed that way. Relabel outputs only.
  - Add `resolution.36e5a8af7000.json` (77a2b062) and `side22_classify.py` to the fail-closed list. Check uncommitted changes as well as commits. Step 10's hash line also carries rows_rt.json 0b038f03 and the cross_r check.
  - The fable recompute takes se_diff from the repo's `zero_kg_check`. "Agree" means the same Q, E, tier and pick. Ties are ≤ 2 episodes of 200.
  - The step-3 identity-map bypass is tied to the identity map itself. Also assert the mapped seeds are distinct.
  - The selection row "no stop" means: stop only on a fable disagreement.
  - If RT_s20s fails, stop and ask before step 5.
  - Quarantine a BAD stamp before its re-run (`find()` keeps the earliest).
  - On G PASS with failing suite lines, confirm-2 still runs; its rows inform the HALT_R decision.
  - Optional report-only drift at the real iterations past TDR (510 / 1010) for an R HALT.

## Out of scope / later

- Mix or reward levers: the next arm on HALT_R (yaw).
- Priced overshoot and the per-kind ladder clamp (code).
- P3 perception and P4 lost-foothold levers.
- Re-reading es3 or es4; es9+ is reserved for later arms.