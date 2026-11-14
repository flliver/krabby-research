
# Bake plan: rough-terrain pairing as the omni policy of record

**Status: PROPOSED, awaiting your decision.** Nothing is baked, committed, pushed or merged. The P2 read routed
USER_DECIDES (CHARTER "2026-11-13 16:00 — P2 step 8"). The stand re-pair finished at 22:58 (CHARTER "2026-11-13 23:30 —
P2 step 9"): SN and SH are both HALT on lines that re-anchor under the pre-registered rules, with nothing remaining.

- **Template:** the rod-speed DR (domain randomization) bake: `2026-11-07_rod_speed_dr/BAKE_PLAN.md`, commits 825ea251
  (record), 2ceff431 (bake) and 0a2617d6 (CI test fix). P2.md steps 11–12.
- **Shas:** c1, C1S, SN and SH were re-hashed on disk on 2026-11-13. The ONNX files were re-hashed in the main checkout.
- **Pairing checks still to run (CPU, at bake time):** C1S's teacher weights equal c1's, bit for bit. SN and SH name
  c1 as their walker and carry its estimator. SH's warm start is SN 5bedc50e.

## 1. What goes in

| Role | New dir (proposed) | File | sha256 | Size | How |
|---|---|---|---|---|---|
| Walker of record, c1 | `rt_11_p3b_rough/` | `model_29500.pt` | `6d1f29be…d3c1` | 12.6 MB | bundled |
| Depth student, C1S | `rt_12_student/` | `policy.onnx`, `depth_latest.onnx`, `omni_student_io.json` | `86d1c801…`, `cca6952d…`, `4ee03d02…` | 36.7 MB | exports bundled |
| | | `model_34499.pt` | `6b48ef33…36a1` | 122 MB | cited by sha (over the 100 MiB guard, like c31000) |
| Stand net, SH | `rt_stand_net/` | `model_5000.pt` | `6f8b558e…4e4c` | 6.1 MB | bundled |
| SH's warm-start ancestor, SN | `rt_stand_net/` | `model_4000.pt` | `5bedc50e…d6a3` | 6.1 MB | bundled |

- **Sources.**
  - c1 is RT_s19 `model_29500`: TDR plus 509 iterations on `RT_train.env`.
  - C1S is RT_ST_s19 `model_34499`, the selection pick (`results/p2/sel.json`).
  - The ONNX files come from `parkour/outputs/omni_plan/rough_terrain_p2/onnx/PICK_6b48ef33/` (main checkout only).
    Their parity is 1.16e-6 against a tolerance of 1e-4 (`results/p2/onnx_PICK.json`).
  - SN is `SN_RT_6d1f29be_sn_s19` (6,097,610 B). SH is `SH_RT_6d1f29be_sh_s19` (6,102,474 B; DAgger, 5000
    iterations, deadband 0.25, trim branch, warm start SN 5bedc50e; `results/p2/stand/sh_head.json`).
- **New blobs:** about 61.5 MB, the same as the rod-speed DR bake. `omni_student_io.json` is byte-identical to
  c31000's and F19's.
- **Nothing is deleted.**
  - TDR (`rsd_9_p3b_rodspeed_dr`) stays a live stage, as c1's resume point.
  - `rsd_10_student/` and `rsd_stand_net/` become superseded folders, with their files kept in place.
  - Stages go from 10 to 11, kept files from 13 to 18, and tracked policy files from 30 to 36.
- **Main since this branch's base (3f944c4b):** 11 commits and 439 files: firmware, the hardware layer (HAL), docs and
  tests. None of them is under `parkour/`, and main's code_sha is still de3772e2. `docs/JETSON_DEPLOYMENT.md` changed
  on both sides, but in different hunks (main at lines 287–302, the bake near line 357). Check it at the PR.
- **This branch's own code (disclosed; not touched by the bake).** The branch changes 167 files against 3f944c4b
  (recounted 2026-11-14 with the terrain-video commit): 135 in the campaign dir and 32 outside it. 31 of the 32 come
  from commits 6588f1c3 and 82946834, all default-off. The 32nd is `experiments/tools/record_categories.py`, the
  terrain-video recorder's `--task` / `--policy-role` options: default-off, under neither hash, not Task 4-locked.
  16 of the 31 are under
  the code pathspecs (the site terrains, `rt_friction.py`, the omni knobs, `observations.py`). One is the Task 4-locked
  `experiments/tools/run_candidates.py` (+138 / −7). So **merging** this branch moves main's code_sha from de3772e2 to
  89c1916b and needs a Task 4 relock plus a CI-faithful simulation first, as P2.md "Out of scope" says. The bake
  commits themselves touch neither hash nor the lock. Until the merge, the rough-terrain kind rows replay only on this
  branch's code. Default: bake on this branch, as the rod-speed DR bake did; the merge is your PR.

## 2. The play / eval / deploy contract (it differs from the rod-speed DR bake's)

1. **Stock rows keep the 27-key env.** This covers suite, field, ring, walkstop and the stand rows.
   - REFS(c1) suite@1 (`phase2/lr/REFS_RT_6d1f29be/.../2026-11-13_21-02-53`) carries exactly the manifest's
     `eval_env` with CLOCK_HIDE=1. That env equals REFS(TDR)'s.
   - The C1S read rows (`P2R_6b48ef33_R_suite`) carry the same 27 keys. N's field rows add only their field block.
2. **The friction change is silent but neutral on stock rows.**
   - With `KRABBY_RT_FRICTION` unset, the privileged friction input is the raw shape μ, b (`observations.py:69, 170-171`).
   - On c1's non-gravel training envs it was 2·((b+1)/2) − 1 = b (`rt_friction.py:12, 87-89`): the same value, the
     same distribution and the same physics.
   - Step-5 neutrality: TDR is BITWISE equal on the new code (`results/step5/neutrality.json`).
3. **Rough-terrain rows need three things the stock contract does not have.**
   - This branch's code (code_sha 89c1916b). Main's code allows only `KRABBY_FLAT_TERRAIN_GEOM` ring or field
     (3f944c4b `omni_terrain_knobs.py:102`).
   - The row's own env: `manifests/envs/rt_kind_eval.env` (26 keys: the 27 minus GEOM) plus the row's block from
     `manifests/scenarios_rt.yaml`, copied verbatim, not rebuilt from this list. The recorded kind rows carry 7 keys
     on top: GEOM site, SITE_LADDER rt, SITE_MIX `<kind>=20`, the SITE_DIFF band, FLAT_TERRAIN_DIFF 0.20:1.0,
     FLAT_TERRAIN_FLAT_FRAC 0.0 and FLAT_TERRAIN_MODE light (`P2R_6b48ef33_rt_slopes_b2` run_meta: 33 keys).
   - On **gravel rows only**, `KRABBY_RT_FRICTION=gravel:0.2:0.5` (`PLAN.md:116-118`). Leaving it off is SILENT: the
     row then plays rigid μ ≥ 0.8 ground.
4. **The student's deploy is unchanged.**
   - `OmniStudentObsPrep` pins the tile flags to the field pair (1, 0). It also overwrites columns 207–215, which
     hold the friction value, with the estimator's output (`omni_student_step.py:12-21, 62-66`).
   - So no rough-terrain input reaches the deployed student. A Jetson host only swaps the two ONNX files.
5. **What the manifest must carry:**
   - The `hazard` text: GEOM may be `site` for kind rows only.
   - `KRABBY_RT_FRICTION` listed as a training and gravel-eval knob, never part of the play env.
   - A kind-row env block naming items 3a–3c.
   - `play_notes` for c1 and C1S that name `--policy-role student|teacher` on Student-Play.
   - The stand `evaluate.env` built from `results/p2/stand/walker_spec.json`, made `$REPO`-relative, with REFS_NAME
     `REFS_RT_6d1f29be`.
   - `KRABBY_DR_ROD_SPEED`, `KRABBY_SITE_*` and `KRABBY_YAW_W` stay out of the play env.

## 3. Defaults the approval covers

1. **Names.** `rt_11_p3b_rough` (stage `p3b_rt`), `rt_12_student` and `rt_stand_net`. `current` and `heads.payload`
   point at c1, `heads.student` at `rt_12_student`, and `heads.stand` at `rt_stand_net`.
2. **New `superseded` entries:**
   - `RSD`, the pairing record, with no files (TDR stays live);
   - `RSD_student`, the c31000 block moved verbatim, with its 3 files kept;
   - `SH_RSD_TDR` and `SN_RSD_TDR`, with `rsd_stand_net/` kept.
3. **Two local commits, never pushed, every file staged by name.**
   - **Commit A, the record:**
     - the 22 untracked `results/p2/stand/` files: the 9 SN-era files (`walker_spec`, `body_height_measured`,
       `sn_premise`, `sn_head`, `sn_calib(.md)`, `sn_bridge`, `gates/sn.json`, `gates/sn_reanchor.json`) and the 13
       SH files (`sh_head`, `sh_premise`, `sh_read(.md)` and its spec, `sh_lazy_rh_read(.md)` and its spec,
       `sh_k5_calib(.md)`, `sh_table`, `gates/sh.json`, `gates/sh_reanchor.json`). No `sh_decision.json` was written,
       because no K4 line failed. RSD tracks the same set, except that it also has `gates/sh_decision.json` (its K4
       line failed) and has no `body_height_measured.json`;
     - the CHARTER entries, this plan, and the REPORT addendum (a new file: the campaign has no REPORT.md).
     - Never `.isaac.lock`.
   - **Commit B, the bake:** the manifest, the regenerated policy README, the 6 new policy files, the rewritten
     `tests/unit/test_policy_of_record_omni.py`, the item 5 docs, and the CHARTER "BAKE DONE" entry, the REPORT
     status line and this plan's status line (as in 2ceff431).
4. **Protect the worktree-only runs.** Hardlink-copy `crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19` (861 MB) and
   `crab_hex_omni_student/2026-11-12_15-08-00_RT_ST_s19` (4.6 GB) into the main checkout's `parkour/logs/rsl_rl/`.
   - The main checkout has neither today, and both `.pt` link counts are 1. Both trees are on one filesystem, so
     this uses no extra disk.
   - The copy keeps the only copy of `model_34499.pt` safe, and the `source:` paths then resolve in both checkouts.
5. **Living docs:**
   - the root README line 145;
   - `docs/FOLDER_LAYOUT.md`, `GLOSSARY.md` (adding c1, C1S, the site geometry and the friction contract),
     `JETSON_DEPLOYMENT.md` and `RUNTIME_ARCHITECTURE.md`;
   - the omni README;
   - the `bundle_policy.py` docstring and the manifest's header list of consumers.

   No `rsd_lib.py`-style fallback is needed (§3 of the test notes).
6. **Disclosures the manifest must carry:**
   - **The USER_DECIDES read:**
     - T PARTIAL: ledge_down +0.171 is a gain, but slopes +0.094 misses the 0.10 floor;
     - T-reg ledge_up −0.055 FAIL, against a 0.05 margin;
     - Tr: track_err diag at es9 is over its line by 0.00076 (≈ 0.0008), a predicted cost of c1;
     - N PASS, and no E flag.
   - **The R-tracking overrule at HALT_R** (user 17:25 "A"): track_err back fails at es5 and es6, and yaw_progress
     turn fails at es6. The costs are back overshoot 1.08× and yaw −3.5 %.
   - **The SN re-anchor:** 31 / 35 decisive lines pass, and the 4 G5 body-height lines are re-anchored on R / HOLD
     (`gates/sn_reanchor.json`, remaining []).
   - **The SH re-anchor:** 94 / 101 decisive lines pass (`gates/sh.json`, `gates/sh_reanchor.json`, remaining []).
     - The 4 K5.G5 body-height lines re-anchor on R / HOLD (head − control ≤ 0.0025 m against 0.03).
     - The 3 K6 lines (homed share bin 1 at es1 and es2, settled share bin 1 at es2) miss the S1-era line by 0.0007,
       0.017 and 0.001. The lazy R_h beside c1 fails the same lines (0.62 / 0.72 / 0.72), so they re-anchor on it
       (lines 0.498 / 0.609 / 0.612; head 0.84 / 0.88 / 0.90).
     - K4 passes: restore misses 1 and 2 against the gate line 3. The reader's own K4 column uses this read's R_h
       (1 miss) as the line and marks es2's 2 as over. The lazy R_h beside c1 misses 5 and 6.
     - Against the baked RSD stand, SH's 7 failed lines are a strict subset of RSD's 12, and SH is closer on each
       of them. SH clears K4 and 3 of RSD's 6 failed K6 lines.
     - **Some passing lines are worse than RSD's.** The walk-stop median settle is 0.6 to 1.1 s slower, with thin margins:
       SN 3.80 / 3.89 s and SH 3.74 / 3.64 s, against lines 3.92 / 3.96 and RSD's 2.70 / 2.80 (SN) and 2.86 / 3.00
       (SH); the margins are 0.12 / 0.07 s (SN) and 0.18 / 0.32 s (SH). SN's field_turn stand-fall rate is 0.01, against a line of 0.020 (RSD 0). SH's bin-2 homed and settled
       shares at es1 are 0.97 against 0.93 (RSD 1.00).
     - The verifiers also found that the R and HOLD body-height controls are the S1-era rows RSD used, not rows run
       beside c1. The 11:53 rule names exactly those rows. Their spare beyond the 0.03 line is 0.011 m, against
       RSD's 0.025 m.
   - **Seeds:** one training seed for every network. Seed-20 replication failed, and c1 is an early checkpoint (+509 of
     6000 iterations) picked from 4 candidates.
   - **Gravel is rigid** and low-μ, not loose ground (P5 follow-on).
   - **back_overshoot reopened** as a follow-up.
   - The stand nets were trained on the wall field and gated at 0 kg (suite, walkstop and table rows on the flat
     ring, field_hard and field_turn on the wall field), never on the rough-terrain kinds.
   - **Camera use is UNVERIFIED** for C1S.
7. **Not included unless you ask:** Walk-Play rows for c1 beyond REFS(c1), a local tag, and a campaign
   `deliverable.json`.

## 4. Steps and checks

1. **Baseline (read-only).**
   - `bundle_policy.py --check` reads 10 stages (1 cited, not bundled: `rsd_10_student`) + stand network
     `rsd_stand_net` (+ its warm-start ancestor) + 13 kept files, current `rsd_9_p3b_rodspeed_dr`: OK. It read OK on
     2026-11-13.
   - Record the unit-test failure set.
2. **SH: done.** The SH values above are filled in. No K4 line failed, so the chance rule (settled 6f) was not needed.
   The 7 failed lines re-anchor under the body-height rule (settled 6e) and the lazy-control rule.
3. **Hardlink copy** (item 4). Re-hash both `.pt` files in the main checkout; the link count must be 2.
4. **Commit A** (item 3). Afterwards `git status` shows only the lock, and `test_experiment_layout.py` passes.
5. **Pre-place the main-checkout-only files** with `cp -p`, then re-hash: the 3 ONNX / io files, SN and SH. RSD step 4
   is the precedent.
6. **Manifest edits.**
   - A 2026-11 BAKE header and new `current` / `heads` / `baked` / `recipe` text.
   - The c1 block: `overlay_delta` gives the PLANNED_SET changes (`rt_machinery.py`); `env_file envs/RT_train.env`;
     `run_record results/runs/RT_s19.json`.
   - c1's gate: `short_arm/gates_fresh.json` HALT_R, overruled by the user.
   - The C1S block: `distill.env_file envs/P2_distill.env`.
   - The stand block.
   - The new `superseded` entries, checked verbatim against the current manifest.
   - TDR's notes no longer say it ships.
   - The §2 contract text.
7. **`bundle_policy.py --sync`, then `--check`.** It should print: 11 stages (1 cited, not bundled: `rt_12_student`)
   + stand network `rt_stand_net` (+ warm-start ancestor `model_4000.pt`) + 18 kept superseded files, current
   `rt_11_p3b_rough`: OK. Also check:
   - no `*.pt` in `rt_12_student/`;
   - the staged set is only the manifest, the README and the 6 files.
8. **Rewrite the policy tests.** Every rod-speed DR check keeps a counterpart, either live or as a superseded check.
   - The stamp-env test adds the kind rows under their own env.
   - The kept-paths test gains `p2_onnx.py` and RSD's `bake_videos.sh` as users.
   - Run the policy, campaign (RSD and RT), layout, stand and student suites, then the whole unit suite. Its failure
     set must equal step 1's.
   - Run the CI-faithful simulation before calling CI green.
9. **Prove nothing pinned moved.** No staged path is in `pipeline.lock.json`, and the code / eval pathspec diffs of
   commit B are empty.
10. **Records.** A CHARTER "BAKE DONE" entry, the REPORT status line and memory. Then commit B, local only, and
    notify. You publish through your PR.

**Execution:** as with the rod-speed DR bake: an opus build, then two independent sonnet·high verifiers, with the
main session staging and committing. No GPU is needed: the videos are P2 step 10 and run before you decide.
