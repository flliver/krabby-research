# Bake plan — rod-speed DR pairing as the omni policy of record

**Status: APPROVED 2026-11-09 23:08 ("Bake as proposed", CHARTER §6), with the §2 defaults. Executing; nothing is
pushed.** F19 stays the policy of record in `policy/` until commit B lands.

- **Source:** a read-only map of the F19 bake (de81f98a; also the stand bakes 4f3019b2 / 75d6261b and the docs pass
  593a047e), and an independent check of that map. Workflow `rsd-stand-addendum-and-bake-map`, 2026-11-09.
- **The check:** no blocking problem. Its two important fixes and seven minor ones are folded in below.
- **Shas:** every sha and size was re-hashed on disk.
- **Pairing checks (CPU):**
  - c31000's teacher weights equal TDR's, bit for bit.
  - Both stand nets name TDR as their walker and carry its estimator.
  - SH's warm start is SN 6cd82ff5.

## 1. What goes in

| Role | New dir (proposed) | File | sha256 | Size | How |
|---|---|---|---|---|---|
| Walker of record, TDR | `rsd_9_p3b_rodspeed_dr/` | `model_28991.pt` | `03c3e472…8567` | 12.6 MB | bundled |
| Depth student, c31000 | `rsd_10_student/` | `policy.onnx`, `depth_latest.onnx`, `omni_student_io.json` | `ac0ea3e2…`, `0a04e065…`, `4ee03d02…` | 36.7 MB | exports bundled |
| | | `model_31000.pt` | `155b5981…e634e` | 122 MB | cited by sha (over the 100 MiB guard, as F19's) |
| Stand net, SH | `rsd_stand_net/` | `model_5000.pt` | `89f4428b…dd8d` | 6.1 MB | bundled |
| SH's warm-start ancestor, SN | `rsd_stand_net/` | `model_4000.pt` | `6cd82ff5…534d` | 6.1 MB | bundled |

- **New blobs:** ≈ 61.5 MB; F19 added 150.5 MB. `omni_student_io.json` is byte-identical to F19's, so the ONNX deploy
  interface is unchanged.
- **Nothing is deleted:**
  - F19's s1…p3b stay live stages, with T* as TDR's resume point.
  - `f19_9_student/` and `f19_stand_net/` become superseded files, kept in place.
  - The kept-file count goes from 8 to 13, and the stage count from 9 to 10.
- **The play / eval / deploy contract is F19's.**
  - It keeps the same 27-key env. The REFS(TDR) suite@1 stamp carries exactly those keys.
  - `KRABBY_DR_ROD_SPEED` is training-only and never set in play, eval or deploy.
  - A Jetson host only swaps the two ONNX files.
- **Pins:**
  - The bake touches no file in Task 4's `pipeline.lock.json` and nothing under the code or eval pathspecs.
  - The live code_sha (de3772e2) and eval_sha (9fe964d7) do not change, and no relock is needed.
  - Task 4's lock (code 6a4e4c6e) was already behind the live code_sha before the bake.
- **Main since this branch's base (ab390e6f):** 184 changed files, none of them touched by the bake (firmware, HAL,
  images and CI only). No conflict is expected at the PR.

## 2. Defaults the approval covers

1. **Names.** `rsd_9_p3b_rodspeed_dr`, `rsd_10_student` and `rsd_stand_net`, continuing F19's numbering under a new
   prefix. `heads.payload` and `current` point at TDR.
2. **Two local commits, never pushed, every file staged by name.**
   - **Commit A, the record.** STAND_REPAIR.md, this plan, `scripts/rsd_stand.py`, `scripts/rsd_stand_queue.sh`,
     `results/stand/` (34 files), and the CHARTER and REPORT edits since 2a07ebeb, which include the REPORT's §13 stand
     addendum. Those diffs are read in full before staging.
   - **Commit B, the bake.** The manifest, the regenerated policy README, the six new policy files and the rewritten
     `tests/unit/test_policy_of_record_omni.py`, plus the item 4 fix and the item 5 docs.
3. **Protect the worktree-only runs** (REPORT §11 decision 8). Hardlink-copy the TDR, TCTL and student run dirs into
   the main checkout's gitignored `parkour/logs/rsl_rl/`, with no extra disk. This keeps the only copy of c31000's
   `.pt` safe and lets the manifest's `source:` paths resolve in both checkouts.
4. **A small fallback in `scripts/rsd_lib.py`** (`manifest_student_sha`, lines 329–335). It looks for F19's student
   under `superseded`, so the campaign tests don't break once `f19_9_student` leaves the stage list. The file is
   unpinned.
5. **Living docs updated in commit B:**
   - the omni README;
   - `docs/FOLDER_LAYOUT.md`, `docs/GLOSSARY.md`, `docs/JETSON_DEPLOYMENT.md` and `docs/RUNTIME_ARCHITECTURE.md`;
   - the root README line 145;
   - the `bundle_policy.py` docstring.

   Stale prose inside hash-pinned files stays for a milestone doc pass with your hash acceptance: `sh_core.py`, the
   stand scripts, `run_phases.py:15` and `crab_hex_omni_env_cfg.py:480`.
6. **Disclosure in the manifest.** The stand block records `gates/sh.json` HALT, the ten re-anchored lines and the K4
   acceptance (`gates/sh_decision.json`) with its reason. Its caveats and follow-ups carry:
   - K4 5 / 6 against 3;
   - walkstop homing 0.92 / 0.91;
   - bin-1 homing 0.74 / 0.81;
   - a single training seed;
   - the uninformative exit-fall reading;
   - the reader mismatches: two K5-control INVALID cells.
7. **Not included unless you ask:**
   - a walker + stand mix video beside TDR (F19 shipped two; about 10–20 min of GPU);
   - Task 4-style Walk-Play rows for TDR (about 1 h of GPU). Without them, TDR's manifest evals are labelled by source:
     REFS(TDR) rows, plus G1 teacher-role cells on Student-Play, with 180 kg never read;
   - a local tag;
   - a campaign `deliverable.json` / `head/` keep-set.

## 3. Steps and checks

1. **Baseline (read-only).**
   - `bundle_policy.py --check` reads OK on F19.
   - Record the full unit-test failure set. The known baseline is the 2 R18 tests until 96028b98 is merged.
2. **Hardlink copy** (item 3). Re-hash `model_28991.pt` and `model_31000.pt` in the main checkout; the link count is 2.
3. **Commit A** (item 2). Afterwards `git status` shows only the untracked locks and `k1/` / `k2/`, and
   `test_experiment_layout.py` passes.
4. **Pre-place the stand files.** Their sources live only in the main checkout, so `--sync` there could only verify.
   `cp -p` both into `policy/rsd_stand_net/` and re-hash.
5. **Manifest edits:**
   - **Header and records.** A 2026-11 BAKE header paragraph, and new `current` / `heads` / `baked` / `recipe` entries.
   - **Moved verbatim into `superseded`.** The F19 student block becomes `F19_student` and the F19 stand block
     `SH_T4F19`. The verbatim check runs against 64f397e5, which is already on main.
   - **New entries.** New TDR, student and stand blocks; new superseded entries `F19`, `F19_student`, `SH_T4F19` and
     `SN_T4F19`; follow-ups revised.
   - **Fixes from the check:**
     - the stand block's `SN_WALKER_SPEC` is the whole `results/stand/walker_spec.json` spec, `$REPO`-relative,
       including `REFS_ENV_STAMP`;
     - the `sh_record_video.sh` command stays in `evaluate.commands`, and only the mp4 references are dropped;
     - the warm start's `used_by` gives the per-family G1–G5 counts;
     - the student's `play_notes` and the T* stage's notes name TDR / c31000 and no longer say T* ships.
6. **`bundle_policy.py --sync`, then `--check`.** It should print: 10 stages (1 cited, not bundled: `rsd_10_student`)
   + stand network `rsd_stand_net` + 13 kept files, current `rsd_9_p3b_rodspeed_dr`: OK. Also check:
   - no `*.pt` in `rsd_10_student/`;
   - the staged set is only the manifest, the README and the six files.
7. **Rewrite the policy tests.** Every F19 check keeps a counterpart, either as a live stage or a superseded check.
   - The gate-table test asserts the new reads' two K5-control mismatch strings and the failed lists: lazy `[K4, K5]`,
     original `[K4, K5, K6]`.
   - Run the policy, campaign, layout, stand and Task 4 report / student suites, then the whole unit suite. Its failure
     set must equal step 1's.
   - Before calling CI green, run the CI-faithful simulation (merge ref, `env -i`, uid 1001).
8. **Records.** A CHARTER §6 "BAKE DONE" entry, the REPORT status line and memory.
9. **Prove nothing pinned moved.** No staged path is in `pipeline.lock.json`, and the code / eval pathspec diffs are
   empty.
10. **Commit B, local.** `bake(omni): rod-speed DR pairing = omni policy of record …`, then notify. Nothing is pushed;
    you publish through your PR.

**Execution:** as with F19 (an opus build, then two independent verifiers), with the main session staging and
committing. No GPU is needed unless item 7 adds the video or rows.
