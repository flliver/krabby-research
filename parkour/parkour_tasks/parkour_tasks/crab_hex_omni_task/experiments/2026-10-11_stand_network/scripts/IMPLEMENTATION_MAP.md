# Stand network: implementation map (PLAN.md steps 3, 4, A1, A2 loader)

This is a read-only survey for the builders of the `stand_net` launcher variant (step 3), the DAgger trainer and its
A1 `recover-probe` mode (step 4), and the A2 `stand_net_branch` loader. It was written on 2026-10-12 from four
subsystem reads. Each reader's claims were re-checked against the source, and the conflicts between readers are
resolved in §9. Nothing was run on a GPU. Claims marked **(CPU check)** were confirmed with a tiny CPU-only torch or
pure-python run during this survey.

PLAN.md D1–D6 and Amendment 1 remain the rules; this map does not change any of them. Where PLAN leaves a choice to
implementation, this map labels it **OPEN**: the builder decides and records the decision in CHARTER §3.

## Pinned inputs (sha256 verified 2026-10-12)

| file | sha256 | note |
|---|---|---|
| phase-2 launcher `p2r_mode_eval_crab_hex_gait.py` | `59539301…17f3` | last commit 014447d7, 1193 lines. Imported, **never edited** (D4). |
| locked harness `crab_hex_forward_task/scripts/eval_crab_hex_gait.py` | `342339cc…3137` | the launcher's docstring patch points (P2L:58-78) still match it |
| R module `crab_hex_omni_task/mdp/crab_hex_omni_restance.py` | `7a16b8d1…8aa4` | pure torch, imports torch at module top |
| S1 `crab_hex_omni_task/policy/6_p2a_field/model_19994.pt` | `968cc550…c23c` | byte-identical to `parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt`. Its pickle holds `cuda:0` storages (zip `data.pkl` scan). |

## Path aliases used below

- **P2L**: `experiments/2026-10-07_omni_phase2/scripts/p2r_mode_eval_crab_hex_gait.py`
- **H**: `crab_hex_forward_task/scripts/eval_crab_hex_gait.py`
- **RST**: `crab_hex_omni_task/mdp/crab_hex_omni_restance.py`
- **CMD**: `crab_hex_omni_task/mdp/crab_hex_omni_command.py`
- **ACE**: `parkour/scripts/rsl_rl/modules/actor_critic_with_encoder.py`
- **CACE**: `parkour/scripts/rsl_rl/modules/crab_actor_critic_with_encoder.py`
- **OPR**: `parkour/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py`
- **VW**: `parkour/scripts/rsl_rl/vecenv_wrapper.py`
- **ENV**: `parkour/parkour_isaaclab/envs/parkour_manager_based_rl_env.py`
- **OBS**: `crab_hex_forward_task/mdp/observations.py`
- **PH**: `crab_hex_omni_task/config/crab_hex/crab_hex_omni_phases.py`
- **RUN_ARM**: `experiments/2026-10-07_omni_phase2/scripts/run_arm.sh`
- **SNC**: `scripts/sn_calib.py` (this campaign)
- **S1-agent** / **S1-env**: `parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/params/{agent,env}.yaml`

Unprefixed `experiments/…` paths are under `crab_hex_omni_task/`. Unprefixed `crab_hex_*` paths are under
`parkour/parkour_tasks/parkour_tasks/`.

---

## 0. Ground facts that every component relies on

### 0.1 Observation layout vs what S1's actor reads (1149 wide; `run_meta.obs_dim_actual` = 1149)

| cols | real content (env side) | S1 actor, `hist_encoding=True` |
|---|---|---|
| 0–88 | `obs_buf` (OBS:206-227): 0-2 ang vel ×0.25; 3-4 roll/pitch; 5-12 cmd/steer/env idx; 13-14 planar lin vel ×2; 15-16 clock sin/cos (0 under CLOCK_HIDE; col 15 = stand bit only with STAND_OBS=clock); 17-40 joint pos − default (cams wrapped, OBS:200-203; cams = 18,20,…,28; hip pitch = 29-34); 41-64 joint vel ×0.05; **65-82 last action** = `action_history_buf[:, -1]` (OBS:225); **83-88 contact proxy** | 0-74 = "prop", which goes straight to the backbone and is also the **estimator input**. 75-88 enter the **scan encoder**. |
| 89-196 | heights (108) | scan encoder |
| 197-205 | env priv_explicit (9) | scan encoder (true values, never estimated) |
| 206-258 | env priv_latent (53) | 206 → scan encoder. **207-215 are overwritten by the estimator** (the network's "priv_explicit" slot). 216-256 = the priv-latent slot, read only when `hist_encoding=False`. |
| 259-1148 | 10 real history frames, 89 wide, frame j at 259+89j (RST:110-116). Each is the `obs_buf` before this step's push (OBS:248-252). | `obs[:, -750:]` = 399-1148 viewed as 10×75 (ACE:118-120), **misaligned** with the real frames. **216-398 are never read.** |

Network slicing: `num_prop`=75, `num_scan`=132, `num_priv_explicit`=9, `num_priv_latent`=41, `num_hist`=10
(S1-agent:57-110; crab_hex_forward_task/config/crab_hex/agents/crab_hex_rl_cfg.py:12-20 records that this slicing
is self-consistent between train and eval but is not the real layout).

`if_scan_encode` is True (ACE:31; `scan_encoder_dims` [128,64,32]; S1 has `actor.scan_encoder.{0,2,4}`). Backbone
input = 75 + 32 + 9 + 20 = 136.

### 0.2 The forward path the eval uses (the D1 "one rule")

- The harness runs under `torch.inference_mode()` (H:935). It writes in place
  `obs[:, 207:216] = estimator.inference(obs[:, :75])` and then calls `policy(obs, hist_encoding=True)`
  (H:1027-1030). `estimator.inference` wraps the forward in `torch.no_grad` (estimator.py:31-33).
- The p2r launcher's st2 branch (P2L:593-596) runs
  `st_pol(cat(obs[:, :207], st_est.inference(obs[:, :75]), obs[:, 216:]), hist_encoding=True)` on a **new**
  tensor (P2L:549-553), so the harness obs is never written (test_omni_p2r_launcher.py:655-675). It then takes
  `torch.where(stand, a_st, a_s1)`. Both nets run on **every** row.
- `act_inference(obs, hist_encoding=False, …)` returns the actor mean only (ACE:253-255). **The default is
  `hist_encoding=False`**, so every call must pass `True`.
- `p0, p1 = num_prop+num_scan, +num_priv_explicit` = (207, 216), from the estimator cfg (P2L:939-941). Run_meta
  records them as `priv_cols` [207, 216].

### 0.3 Mask facts (RST:282-338; checked with a CPU run)

- `stand_mask_obs(obs, rows=None, *, inplace=False)` writes 264 columns on a 1149-wide row: last action 65-82 → 0.0
  and proxy 83-88 → -0.5, in the current step and in each real history frame. It returns a clone unless `inplace`.
  `rows` must be an (N,) bool tensor or None.
- Of the 264 columns, 240 are read by the actor on the hist path: 10 in prop (65-74), 14 in the scan slice (75-88)
  and 216 in its history view. The 24 columns of real frame 0 (324-347) are never read. **(CPU check)**
- The estimator reads `obs[:, :75]`, which includes masked cols 65-74, so **estimate first, then mask** (D1).
- R reads only cols 3, 4, 18, 20, …, 28 and 29-34 (RST:221-279). Its outputs are identical on raw, own-estimate and
  masked obs. **(CPU check: `torch.equal` on target and branch)**
- Index/value tensors are cached per (width, device, dtype) in `_STAND_MASK_TENSORS` (RST:313-322). A cache
  entry created under inference mode holds inference tensors, but masking a normal tensor outside inference mode
  still works and is differentiable downstream. **(CPU check)**

### 0.4 Env step order: slot, obs and termination alignment (ENV:95-165)

`process_action` (:97) → physics → `episode_length_buf += 1` (:120) → terminations (:123) → rewards →
`_reset_idx` (:142; sets `episode_length_buf=0` at :293, `command_manager.reset` → resample) →
`command_manager.compute` (:153: `time_left -= dt`, resample expired rows) → interval events (:159, pushes) →
`observation_manager.compute` (:162).

So `cmd.stand_mask()` read right after `env.step` returns is the slot of the command that the returned obs encodes.
Each compute returns a **fresh** obs tensor (observation_manager.py:395 `.clone()`, :433 `torch.cat`); the env keeps
a reference in `self.obs_buf` and does not reuse it across steps.

### 0.5 Action and clip semantics

- `ParkourRslRlVecEnvWrapper.step` runs `torch.clamp(actions, ±clip_actions)` (VW:136). `clip_actions` = 1.0
  (S1-agent:7). The harness records **pre-clip** actions (H:1048 is before H:1071).
- Action term: wrapper-clamped action → term clip (cams ±1; rods ±4.8 under 2a full actions,
  crab_hex_env_cfg.py:272, parkour_actions.py:308) → `_action_history_buf` → applied = buf[-1-delay]
  (parkour_actions.py:455-465).
- Delay is 1 throughout: `action_delay_steps` [1, 1], `delay_update_global_steps` 192000, `use_delay` true
  (S1-env:796-806). The list is consumed by `pop()` on the cfg object (joint_actions.py:29, 40-41).
- Obs cols 65-82 = this step's submitted, clamped action. The applied action is the previous step's.

### 0.6 Checkpoint load contract (OPR:761-804)

- `runner.load(path, load_optimizer=False)` reads `model_state_dict` (strict; ACE:261-263 returns True),
  `estimator_state_dict` (strict), `iter` and `infos`. **All four keys are required.**
- `optimizer_state_dict` and `alg_state` are read only when `load_optimizer=True` (play.py / evaluation.py). A
  missing `alg_state` is rebuilt from `iter` (OPR:806-818).
- Unknown top-level keys are ignored.
- `torch.load` is called with no `map_location`, so tensors land on the device they were saved from.
- S1 keys: `model_state_dict` (std_raw(18), actor.{priv_encoder, history_encoder, scan_encoder, actor_backbone},
  critic.* with `critic.0.weight` (512, 1149)), `estimator_state_dict` (estimator.{0,2,4}: 75→128→64→9),
  `optimizer_state_dict`, `iter`=19994, `alg_state`, `infos`=None.
- `KRABBY_STD_PARAM=soft` is required for a strict load: it creates `std_raw` (CACE:96-101). The init value
  must lie in (0.05, 2.0) (CACE:44).

---

## 1. One shared module: `scripts/sn_common.py` (D1 "one function, called identically")

Both the launcher and the trainer import this module **by path**. It must keep **all torch and R imports inside
its functions**:

- The launcher imports it before Kit starts.
- The harness imports torch only after AppLauncher (H:144-149).
- The R module imports torch at its top (RST:69-71).

The phase-2 launcher keeps every torch import local for the same reason (P2L:301-333, 549-553). A top-level torch
import in the eval process changes process state before Kit. See §7 L6.

```text
RESTANCE_D3 = "lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3"
load_rst()            -> the R module by path (as P2L.load_restance, P2L:193-202), cached
r_params()            -> rst.parse_restance(RESTANCE_D3); assert rst.format_restance(p) == RESTANCE_D3   (CPU check: round-trips;
                         p_up = -0.0625 rad, p_down = -0.05 rad)
stand_net_input(obs, estimator, *, p0, p1, num_prop, mask: bool)
    est = estimator.inference(obs[:, :num_prop])                       # unmasked prop (D1)
    x = torch.cat([obs[:, :p0], est, obs[:, p1:]], dim=1)              # == P2L._clone_with_own_estimate, a NEW tensor
    return load_rst().stand_mask_obs(x, rows=None, inplace=True) if mask else x   # inplace on the fresh cat only
r_label(obs)          -> load_rst().restance_target(obs, r_params())   # (target (N,18) in [-1,1], branch (N,6) int64)
```

`rows=None` masks every row of the clone. Rows outside stand are discarded by `torch.where` in both callers, and
the actor's MLP and conv paths are row-independent. That keeps the call free of the stand vector, so trainer and
launcher make literally the same call (§9 C2). `p0`, `p1` and `num_prop` always come from the estimator cfg
(P2L:939-941 / `train_cfg["estimator"]`), never from hard-coded "real layout" numbers (197:205 would be wrong).

---

## 2. Step 3: launcher variant `stand_net`

### 2.1 File, import, isolation

- File: `scripts/sn_stand_net_eval_crab_hex_gait.py`. The name **must end in `eval_crab_hex_gait.py`** and its
  absolute path must contain no spaces. RUN_ARM:99's `drivers_idle` regex finds evals by that suffix.
- First statement: `sys.dont_write_bytecode = True` (as SNC:70).
- Load P2L **by path into a private module instance**:
  `spec_from_file_location("_sn_p2r", P2/"scripts/p2r_mode_eval_crab_hex_gait.py")` + `exec_module`. Patch only
  that instance. Importing it by name would share the patched object with other importers (SNC:418 references the
  file; the tests load it as `_p2r_launcher`).
- P2L's top level imports no torch (P2L:90-101). `HARNESS` and `_THIS` derive from the file's own path
  (P2L:103-106), so a by-path load resolves them correctly.

### 2.2 What can and cannot be extended (verified)

- There is no registration hook. Module globals are resolved at call time:
  - `P2RMode` (constructor at P2L:985)
  - `_load_extra` (P2L:952)
  - `p2r_block` (P2L:1119)
  - `make_factory` (P2L:1136)
  - `VARIANTS` (P2L:157, 272, 434)

  Rebinding these on the private instance changes behaviour without editing the file. `P2RMode` is used only as
  that constructor; `self.mode: P2RMode | None` (P2L:894) is an unevaluated annotation (`from __future__ import
  annotations`, P2L:90).
- Hard-coded gates:
  - `validate_opts` refuses `--p2r-stand-ckpt` unless mode `st2` (P2L:281-284).
  - The factory builds an extra runner only for `st2` / `probe` (P2L:947) and sets `ctx.specialist` only for
    `st2` (P2L:951-954).
  - `_validate_restance` refuses residual / stand_bank / observe on non-restance modes (P2L:232-239).
- **Therefore run internally as mode `st2`** and relabel on the mode object.

### 2.3 Construction recipe

1. **Own args, before anything else.** Use `argparse(add_help=False, allow_abbrev=False).parse_known_args`:
   - `--stand-ckpt PATH`
   - `--stand-mask {0,1}` (`type=int`, `choices=(0, 1)`)
   - Env fallbacks `SN_STAND_CKPT` / `SN_STAND_MASK`, deliberately not `KRABBY_*`, so `run_meta.env_vars` is
     unchanged. They are needed because RUN_ARM `lr_eval` passes no `EVAL_EXTRA_ARGS` (RUN_ARM:552-553).

   Refuse (pre-Kit, SystemExit):
   - a missing ckpt or mask;
   - any leftover `--stand*` or `--sn*` argument;
   - any `--p2r*` in argv;
   - any `P2R_*` variable in the environment. P2L's fallbacks (P2L:127-128, 165-186) would inject a second mode;
     `P2R_BANK_MAX` is accepted silently;
   - `KRABBY_OMNI_STAND_OBS` != `"0"` (D5);
   - `KRABBY_OMNI_STAND_MASK` / `KRABBY_OMNI_STAND_PROXY` not in (`""`, `"0"`) (SNC:139-152).

2. **Hand off.** Call
   `p2r.main([argv[0], "--p2r-mode", "st2", "--p2r-stand-ckpt", ckpt] + rest, harness=harness, environ=environ)`.
   P2L.main strips and validates (the file must exist, P2L:293-296), sets `sys.argv`, runs the harness via
   `runpy`, patches `g["main"].__globals__`, and calls `simulation_app.close()` in its `finally` (P2L:1165-1189).
   Keep the signature `main(argv=None, *, harness=p2r.HARNESS, environ=None)` so the fake-harness e2e tests apply
   unchanged.

3. **`StandNetMode(p2r.P2RMode)`.** Install it with `p2r.P2RMode = <class bound to the mask flag>` before
   `p2r.main`.
   - `__init__(*a, **k)` calls `super().__init__(*a, **k)`. It receives variant `"st2"` plus
     `stand=(st_policy, st_estimator)` from `_load_extra` (P2L:952-953), which passes P2L:434 and 443. Then set
     `self.variant = "stand_net"` and `self.stand_mask = int(flag)`.
   - **Override `__call__` completely.** With variant `stand_net`, the parent's `__call__` falls into the PROBE
     `else` branch (P2L:599-607): S1 acts everywhere and `_true_obs` / `_bank_add` run (§7 L1).
   - Copy P2L:563-588 verbatim: kwargs refusal, bind check, pass check, lazy `c["stand"]`, `alive` init and the
     `t` bound. Then:
     ```text
     stand = self.rederive_on_reset()
     a_s1  = self.s1_policy(obs, hist_encoding=hist_encoding)          # harness obs, S1 estimate already in place
     st_pol, st_est = self.stand
     x     = sn_common.stand_net_input(obs, st_est, p0=self.p0, p1=self.p1, num_prop=self.num_prop, mask=bool(self.stand_mask))
     a_st  = st_pol(x, hist_encoding=True).to(a_s1.dtype)
     actions = torch.where(stand.unsqueeze(1), a_st, a_s1)
     # extra raws (see 2.4): allocate self._cur.setdefault("rst", {...}) on the pass's first call, then write row t
     c["stand"][t] = stand; c["t"] = t + 1; self.calls += 1; return actions
     ```
   - Never write `obs` (the harness records from it and S1 reads it). The mask runs on the fresh cat only.

4. **Estimator identity (D1 says the stand net's estimator is S1's, frozen).** The launcher runs the stand
   checkpoint's own estimator (P2L:595, 924). The check runs inside `StandNetMode.bind_s1`, which the harness
   calls at `get_inference_policy` (H:728), after `runner.load` (H:710) and before the first step:
   - `torch.load(<harness --checkpoint>, map_location="cpu", weights_only=False)["estimator_state_dict"]`
   - compared key by key with `torch.equal` against `self.stand[1].state_dict()` moved to CPU.

   The wrapper reads `--checkpoint` from `rest` without removing it. Record `estimator_equals_s1`. **Refuse**
   (SystemExit) on a mismatch, so the run fails before any GPU episode. The check is CPU-only and draws no RNG.

5. **`p2r_block` wrapper.** Replace `p2r.p2r_block`, which is resolved at call time (P2L:1119), with
   `block(ctx, stats, *, bank_rel, rst_stats=None, stand_bank_rel=None)`. It calls the original, then:
   - asserts `b["variant"] == "stand_net"` (it comes from `mode.variant`, P2L:1043);
   - sets `b["stand_mask"] = int(flag)`. Use an **int, never a bool**: `stand_net_arm` treats a bool as malformed
     (SNC:155-166, the bool test at :164);
   - adds `b["stand_net"] = {wrapper_sha256: sha256(__file__), common_sha256, restance_module_sha256,
     mask_fn: "crab_hex_omni_restance.stand_mask_obs", mask_rows: "all", estimator_equals_s1, r_knob: RESTANCE_D3}`;
   - extends `b["raw_series"]` with the extra keys.

   `specialist {path, sha256}` already comes from `ctx.specialist` (P2L:954, 1048). The block's
   `launcher_sha256` is always the **phase-2 file's** sha (P2L:1045), which is why the wrapper's own sha must go in
   the extra key.

### 2.4 Extra raw series

`_close_pass` writes every key of `c["rst"]` as its own npz key (P2L:507-511). `write_run` checks only
`shape[:2] == (T, episodes)` (P2L:1089-1094).

**Required for the swap-live proof:**

| key | shape and dtype | content |
|---|---|---|
| `sn_stand_action` | (T,N,18) float32 | `where(stand, a_st, 0)` |
| `sn_s1_action` | (T,N,18) float32 | `where(stand, a_s1, 0)`: the counterfactual S1 action on stand rows |

**Recommended now (the step-9 "observe pass", reported per-branch |μ−R|):**

| key | shape and dtype | content |
|---|---|---|
| `sn_r_target` | (T,N,18) float32 | `where(stand, R target, 0)` |
| `sn_r_branch` | (T,N,6) int8 | R branch on stand rows, -1 elsewhere |

Both come from `sn_common.r_label(obs)`, a pure function: no RNG, and actions are unchanged. `p2_read.p2r_bitwise_diff`
compares only keys present in REFS (p2_read.py:2815-2838), so the extra keys never break the identity diff.
`p2r_stand_mode` is still written (P2L:507), which the G3 exit clause needs (SNC:268+, p2_read.py:2901-2946).

### 2.5 Launch (RUN_ARM `--row-eval`, as `sn_calib_rows.sh` does)

```text
P2_GUARD_REF=REFH_S1_s18 P2_EVAL=<abs>/sn_stand_net_eval_crab_hex_gait.py \
EVAL_EXTRA_ARGS='--stand-ckpt <abs path, no spaces/globs> --stand-mask <0|1>' \
bash $P2/scripts/run_arm.sh --row-eval <NAME> <S1 ckpt> <REFS_S1_s18 env stamp> <manifest> <scenario> <es>
```

- **Env stamp:** `parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24`. It
  carries `KRABBY_OMNI_STAND_OBS='0'`, CLOCK_HIDE=1, STD_PARAM soft, INIT_STD 0.8 and ellipse 0.35:0.35:0.25.
- `row_eval` applies the stamp's `env_vars`, then `P2_ENV_EXTRA` (RUN_ARM:529-532). **Do not** put `STAND_OBS` in
  `P2_ENV_EXTRA`.
- `EVAL_EXTRA_ARGS` is unquoted (RUN_ARM:532).
- Manifests, as the phase-2 queues launched REFS (p2s_queue.sh:53-57, 98-111):

  | rows | manifest |
  |---|---|
  | `suite__omni` | `experiments/eval/scenarios_omni.yaml` |
  | `field_hard__omni`, `field_turn__omni` | `experiments/2026-10-07_omni_phase2/scenarios_p2.yaml` |
  | `walkstop__omni` | `experiments/2026-10-07_omni_phase2/scenarios_p2s.yaml` |

- Hashes: campaign scripts and repo `tests/` are dropped from `code_sha`, `eval_sha` and `dirty`
  (`experiments/tools/run_candidates.py:197`, `CODE_DROP_GLOBS`).

### 2.6 Identity bridge and swap-live proof

- **Identity bridge (step 3).** Run `--stand-ckpt <S1 file> --stand-mask 0` on suite, walkstop and field_hard
  es1. Then `p2_read.p2r_bitwise_diff(REFS_stamp, SN_stamp)` must return `[]`. Expected mechanism: the stand
  input `cat(obs[:, :207], est_S1, obs[:, 216:])` is bitwise the harness obs after its in-place write, the weights
  are identical, and the batch is the same, so `a_st == a_s1`. Only the GPU run settles it (§8 U1).
- **Swap-live**, a campaign-local reader modelled on `restance_swap_live` (P2L:828-886). For each raw episode,
  with `mode = p2r_stand_mode`:
  - (a) `actions[mode] == sn_stand_action[mode]` bitwise and `sn_stand_action[~mode] == 0`;
  - (b) stand share > 0;
  - (c) `actions[:first_entry]` equals REFS on the same episode index (as P2L:873-880);
  - (d) for a non-identity checkpoint, some stand step has `sn_stand_action != sn_s1_action`; for the identity
    bridge, they are all equal.

  Also require `p2r_mode.variant == "stand_net"` and `stand_mask` an int in {0, 1}.

### 2.7 Unit tests (CPU; repo `tests/unit/`, which is outside the hashes)

Reuse `tests/unit/test_omni_p2r_launcher.py`: `STUBS` (:59), `FAKE_HARNESS` (:292), `_run_launcher` (:794-801),
the `ckpts` fixture (:808-812) and the st2 pattern (:655-675). Assert:

1. argv handling: the harness argv order is kept; `--stand*` leftovers, `--p2r*` and `P2R_*` are refused;
   `STAND_OBS != "0"` is refused pre-harness.
2. `StandNetMode` routes `a_st` only on stand rows. The harness obs is unchanged. The clone carries the stand
   estimator's output in cols 207:216. With mask 1, the clone equals `RST.stand_mask_obs` applied to
   (estimate-then-cat).
3. **Launcher mask == trainer mask:** record the input a stub stand policy receives in `StandNetMode.__call__` and
   in the trainer's routing function on the same obs and estimator. They must be bitwise equal on stand rows.
4. e2e: `p2r_mode.variant == "stand_net"`; `stand_mask` is an `int`; `specialist.sha256 == sha256(ckpt)`;
   `estimator_equals_s1` is present; the extra raw keys are present with the right shapes.
5. Stand ckpt == S1 with mask 0 gives actions bitwise equal to the unpatched fake run.
6. A stand-ckpt estimator that differs from S1's is refused.
7. `SNC.stand_net_reasons(…)` returns `[]` on the stub stamp's run_meta, given an env with `STAND_OBS="0"`.
8. The existing `test_launcher_name_matches_run_arm_idle_check` idea, applied to the new filename.

---

## 3. Step 4: the DAgger trainer (`scripts/sn_dagger.py`; modes `dagger` and `recover_probe`)

### 3.1 Process environment (set before Python starts)

Writing to the environment after Kit starts changes process state (memory: eval-process-state-sensitivity). Preset
expansion also runs at import with `setdefault` (`crab_hex_omni_task/config/crab_hex/__init__.py:8-10`;
PH:346-381, with the setdefault at :378-380).

- **Start from a clean base:** `os.environ` with every `KRABBY_*` removed, plus `OMNI_KIT_ACCEPT_EULA=yes`,
  `TERM=xterm` and `PYTHONUNBUFFERED=1` (as `run_phases._subenv`, experiments/tools/run_phases.py:1140-1148).
- **Preset and plant:** `KRABBY_OMNI_PHASE=2a`, `KRABBY_PLANT=A15+B`.
- **Overlay:** every key of the `P2__S1_s18` overlay verbatim (experiments/2026-10-07_omni_phase2/candidates.yaml:551-576).
  This includes the **empty-string** `KRABBY_PHASEOUT=''` and `KRABBY_OMNI_DIR_BAND=''`, both present.
- **Changes the PLAN asks for**, recorded as deviations in CHARTER §3:
  - `KRABBY_OMNI_CMD_POOL='stand:0.85,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07'`
  - `KRABBY_OMNI_SLOTS='0.85:0.55:0.1:0.2'`

    The pool projects to (0.5, 0.3235, 0.0588, 0.1176), which matches these SLOTS with deviation 0.0 against
    `POOL_SLOT_MATCH_TOL` 1e-6 (CMD:301-320). It keeps S1's within-slot type proportions. **(CPU check:
    `parse_cmd_pool` and `projected_slot_probs` loaded by path)**
  - `KRABBY_RESAMPLE_S='3:12'`: `apply_flat_walk_knobs` sets `resampling_time_range` (crab_hex_env_cfg.py:528-531).
    It is reached through the 2a teacher-mode branch (:479). Omni code does not override it.
  - `KRABBY_OMNI_STAND_OBS='0'` (D5; it parses to off, the same as absent, motion_knobs.py:491-500).
- **Never set** `KRABBY_PHASE`, `KRABBY_SIDE_PHASE` (PH:365-371 raises), `KRABBY_HEX_USD_PATH`,
  `KRABBY_OMNI_STAND_MASK`, `KRABBY_OMNI_STAND_PROXY` or `KRABBY_RESTANCE`. R's constants live in `sn_common`, not
  in env knobs.
- **Expected expansion** (pure-python `phase_env("2a","A15+B")` + overlay, compared with the REFS walkstop stamp):

  | variable | trainer | eval stamp | effect |
  |---|---|---|---|
  | `KRABBY_FLAT_TERRAIN_FLAT_FRAC` | 0.5 | 0.0 | terrain |
  | `KRABBY_FLAT_TERRAIN_DIFF` | 0.20:1.0 | 0.70:1.0 (field_hard) | terrain |
  | `KRABBY_TRACK_SIGMA2` / `KRABBY_TRACK_L1_W` | 0.04 / −2.0 | 0.1 / −1.0 | rewards |
  | `KRABBY_CLOCK_W` / `KRABBY_APEX_W` | 0.001 | 1.0 | rewards |
  | `KRABBY_DR_PUSH` / `KRABBY_DR_MASS` / `KRABBY_DR_COM` | 0.5 / −0.5:1.5 / 0.01 | (absent) | domain randomization |

  The eval stamps set every `EVAL_CARRY_DEFAULTS` key (run_phases.py:214-223) to its off spelling. S1's own training
  env.yaml shows the same effective obs settings: `facing_obs_encoding` raw, `facing_obs_clip` 0.6, `clock_hide`
  true (S1-env:762-764); `ang_vel_clip` 0.2 (S1-env:2389). So obs-affecting knobs agree by construction. Assert
  this at trainer start; do not rely on it.
- **Launch:** `/home/nickmagus/krabby/isaac_venv/bin/python <abs>/scripts/sn_dagger.py --headless --num_envs 256 --seed <s>`.
  - Use absolute paths throughout.
  - Run under `systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0` (the RUN_ARM:94 `CAP`) and
    `setsid nohup`.
  - RUN_ARM's `drivers_idle` (RUN_ARM:99) does **not** see this filename. Serialize it against evals through the
    campaign's own queue (§7 T14).

### 3.2 Script skeleton (the harness / train.py pattern)

1. Imports in order:
   - `from isaaclab.app import AppLauncher`, then load `cli_args` by path if needed (H:44-50).
   - argparse + `AppLauncher.add_app_launcher_args`, then `app = AppLauncher(args).app` (H:140-145). This comes
     **before** gymnasium, torch or task imports.
   - sys.path surgery exactly as H:152-161: remove `parkour/scripts` and `parkour/scripts/rsl_rl`; insert the
     repo root, `parkour/` and `parkour/parkour_tasks`.
   - `import parkour_tasks` (H:164), `from isaaclab_tasks.utils import parse_env_cfg`,
     `from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry`,
     `from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper`.
   - **Do not import `train.py` or copy its TF32 lines** (train.py:109-112). The harness sets none of them.
2. `env_cfg = parse_env_cfg("Isaac-Crab-Hex-Omni-Teacher-v0", device=args.device, num_envs=256)` (H:631 pattern)
   and `env_cfg.seed = seed`. Registration: `crab_hex_omni_task/config/crab_hex/__init__.py:43-51`. The default
   `num_envs` is 6144 (crab_hex_omni_env_cfg.py:278).
3. `agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")` (cli_args.py:55) and
   `train_cfg = agent_cfg_to_train_dict(agent_cfg)` (runner_factory.py:42-53).
4. **Fail closed before `gym.make`:**
   - `resampling_time_range == (3.0, 12.0)`
   - `cmd_pool ==` the string above, `slot_probs == (0.85, 0.55, 0.1, 0.2)`, `dir_band is None`
   - `episode_length_s == 40`
   - `parkours.base_parkour.move_up_frac / move_down_frac == 0.7 / 0.4`
   - `freeze_terrain_levels` False
   - `agent_cfg.clip_actions == 1.0`
   - `policy.noise_std_type == "soft"`, `init_noise_std == 0.8`
   - `terminations.crab_failure.params`: `limit_angle` 0.5, `contact_force_threshold` 500, `minimum_root_height_z`
     None (S1-env:2314-2342)
   - observation params `clock_hide` true, `facing_obs_clip` 0.6, `facing_obs_encoding` "raw"
   - no `eval_slot_latch` on the command cfg (CMD:472)
5. `env = ParkourRslRlVecEnvWrapper(gym.make(task, cfg=env_cfg, render_mode=None), clip_actions=agent_cfg.clip_actions)`.
   Its `__init__` calls `env.reset()` (VW:46). Assert `env.num_obs == 1149` and `env.num_actions == 18`. Build
   **one env per process**, because `action_delay_steps` is popped from the cfg object (§0.5).
6. Dump `params/env.yaml`, `params/agent.yaml` (as train.py:193-194, `isaaclab.utils.io.dump_yaml`) and the
   `KRABBY_*` environment, to support the D5 parity audit.

### 3.3 Networks: S1, the estimator and the student (no PPO runner)

- **Build** inside `save_rng()` / `restore_rng()` + `rng_equal` (P2L:301-333). Network init draws on the global
  RNG, which the env's command sampler also uses.
  ```text
  pc = deepcopy(train_cfg["policy"]); pc.pop("class_name")      # builders mutate cfg dicts (OPR:139-141, ACE:164, :48, :54)
  ec = deepcopy(train_cfg["estimator"]); ec.pop("class_name")
  s1 = CrabHexActorCriticRMA(1149, 18, **pc).to(dev)             # scripts.rsl_rl.modules.crab_actor_critic_with_encoder
  est = DefaultEstimator(**ec).to(dev)                           # scripts.rsl_rl.modules.feature_extractors
  ck = torch.load(S1, map_location=dev, weights_only=False)
  s1.load_state_dict(ck["model_state_dict"], strict=True); est.load_state_dict(ck["estimator_state_dict"], strict=True)
  s1.eval(); est.eval(); s1.requires_grad_(False); est.requires_grad_(False)
  student = copy.deepcopy(s1); student.requires_grad_(False)
  student.actor.actor_backbone.requires_grad_(True)              # + student.actor.scan_encoder.requires_grad_(True) iff OPEN-1 says so
  ```
  - Use a **fresh deepcopy** of the cfg for each build. Construction pops `class_name` and `channel_size` from
    nested dicts, and a second build from the same dict raises `KeyError`.
  - The constructor refuses unknown policy keys (ACE:153-158).
  - Without `KRABBY_STD_PARAM=soft` the strict load fails: it expects `std` and finds `std_raw`.
- **Why not a factory runner.** `OnPolicyRunnerWithExtractor.__init__` calls `env.get_observations()` (OPR:120),
  which pushes a history frame. It also builds PPOWithExtractor, RolloutStorage and symmetry from the overlay's
  PPO knobs.
  - If a factory runner is wanted anyway, build it on `p2r.StandInEnv(env, template)` (P2L:337-360) with the RNG
    restored, exactly as P2L:947-964 does.
  - `runner.save()` cannot be used outside `learn()`: `logger_type` is set only at OPR:256 / 426, so it raises
    AttributeError.
- **Gradient reach** of the D1 path with the history encoder frozen: only `actor.actor_backbone` and
  `actor.scan_encoder` (runner reader, CPU check). This is consistent with ACE:90-116: `priv_encoder` is used only
  on the `hist_encoding=False` path, and the critic and `std_raw` are not used by `act_inference`.
- **Optimizer:** Adam over **all** `student.parameters()` in named order. Frozen parameters keep `grad=None` and
  Adam skips them. This keeps `optimizer_state_dict` loadable by PPO's single-group Adam
  (ppo_with_extractor.py:421) when play.py or evaluation.py call `runner.load(path)` with `load_optimizer=True`.
  Clip gradients as PPO does (`max_grad_norm` 1.0, S1-agent:31).

### 3.4 Rollout loop: routing, labels and exits (the trainer side of D1/D2/D5)

```text
obs, _ = env.get_observations()            # ONCE at start (VW:101 computes obs = one history push); afterwards only env.step's obs
cmd = env.unwrapped.command_manager.get_term("base_velocity")
stand = cmd.stand_mask().reshape(-1).bool()   # slot_id == SLOT_STAND (CMD:672-680); training: pool draw at resample (CMD:575, :610)
loop, under torch.inference_mode() (as OPR:317 and H:935):
    x_s1 = sn_common.stand_net_input(obs, est, p0=207, p1=216, num_prop=75, mask=False)
    a_s1 = s1.act_inference(x_s1, hist_encoding=True)                     # == the harness's in-place write + policy call
    x_st = sn_common.stand_net_input(obs, est, ..., mask=arm_mask)        # A-mask: 1, A-full: 0
    tgt, br = sn_common.r_label(obs)                                      # R on the visited state, unmasked obs
    a_stand = where(executor_is_R[:, None], tgt, student.act_inference(x_st, hist_encoding=True))   # all N rows, like P2L:593-596
    a_stand = a_stand + sigma_ep[:, None] * randn(N, 18, generator=g_dart)                          # DART, stand rows only (sigma 0 off)
    actions = torch.where(stand[:, None], a_stand, a_s1)
    store rows `stand` -> buffer (x_st, tgt, br, ids, executor, sigma)    # in-place copy into a PRE-ALLOCATED normal tensor
    obs, _, dones, extras = env.step(actions)                             # the wrapper clamps to +-1 (VW:136); never env.unwrapped.step
    stand_next = cmd.stand_mask().reshape(-1).bool()                      # aligned with the new obs (sec. 0.4)
    fell = env.unwrapped.termination_manager.get_term("crab_failure").bool().reshape(-1)   # as H:1089
    timeout = extras["time_outs"].bool()                                  # VW:147
    exit = stand & ~stand_next & ~dones.bool()                            # stand -> non-stand on a live row
    reset_rows = env.unwrapped.episode_length_buf == 0                    # new episode: redraw per-episode beta / sigma / control slice
    stand = stand_next
```

- **Episode draws** use private generators (§3.6):
  - `executor_is_R` comes from β(it) per episode, OR'd with the fixed 10 % R-executed control slice.
  - `sigma_ep` ∈ {0, 0.1, 0.3}.
- Consecutive stand draws (stand → stand at a resample) do **not** change the mode. This matches the launcher's
  latched rule, so one "stand" is a maximal run of `stand` per env.
- **Stand-end statistics** (step 7, A1):
  - `|c|` comes from the **last stand row before the exit**: cols 18, 20, …, 28 of the obs the stand row acted on.
    Never use a row after a reset.
  - homed = every leg has |c| ≤ 0.1 rad.
  - A stand that ends in `dones & fell` scores fall=1, homed=0. One that ends in `timeout` is excluded (cut).
  - Exit falls = `fell` within 100 steps after an exit on the same env, tracked with a per-env countdown, matching
    p2_read's 2-s window (p2_read.py:2901-2946).
- **Inference-mode tensors:** env and inference-mode outputs cannot be saved for backward. Writing them **in place
  into a pre-allocated normal tensor** inside inference mode is allowed, and that buffer then trains normally.
  Feeding the raw inference tensor to a training forward raises. **(CPU check)**
- **Stand-event log** (needed by the step-7 notify reads and A1). Write one record per stand to campaign-run JSONL
  or npz. Fields: iteration, env, episode, entry step, exit/end step, end reason (exit / fall / timeout / reset),
  executor (student / R-β / R-control / spoiler kind), σ, scheduled first-draw length (`cmd.time_left` at entry;
  see OPEN-7), homed at end, max |c| at end, fall in stand, exit fall within 2 s, terrain level
  (`env.unwrapped.parkour_manager.get_term("base_parkour")…terrain_levels`).

### 3.5 Labels, loss, buffer

- **Labels:** R from `sn_common.r_label`, using the D3 constants. The class defaults differ: up_frac 0.6, k_h 0.3,
  trim 0.03 and eps_step 0.10 in the dataclass (RST:119-136) against 0.5, 1, 0.1 and 0.3 in D3.
  `parse_restance("default")` returns the class defaults (RST:193-194). Targets are clamped to [-1, 1] (RST:278).
- **Loss:** MSE between `student.act_inference(x_st_buffered, hist_encoding=True)`, the **unclipped** mean, and
  the target. Fitting the clipped action instead would kill the gradient where μ is beyond ±1. The executed action
  is clipped by the wrapper. State this choice in the trainer header.
- **Buffer:** a capped reservoir that stores the **exact** `act_inference` input (`x_st`, after estimate and mask).
  - Optional compaction: store cols 0:216 and 399:1149 (966 floats, about 3.9 KB per row). Rebuild with zeros at
    216:399 for training; the actor reads history from the **end** (ACE:119), so the tail must end the row.
  - Keep alongside each row: target (18), branch (6) and a LIFT-onset flag (a leg's branch becomes LIFT from
    non-LIFT on consecutive stand rows of one env).
  - Branch balancing is **OPEN-3**. LIFT-onset oversampling is required (D3). The smoke asserts LIFT-onset ≥ 1 %
    after balancing.
- **Schedules** (step 6 asserts the function, it does not need to reach the end): β(it) = 1 → 0 linearly over the
  first third, then 0; the 10 % control slice is fixed for the whole run; per-episode σ. The A1 mode overrides
  these (§4).

### 3.6 RNG discipline

- The command sampler draws types on the **global** torch stream: `sample_slots(None, …)` (CMD:575) and
  `time_left.uniform_` (command_manager.py:184). Pushes and terrain draws do the same.
- Every trainer-side draw must use **private** `torch.Generator(device=dev)` and `numpy.random.default_rng`
  objects seeded from the run seed. That covers β and control episodes, σ, DART noise, reservoir and minibatch
  indices, A1 spoiler choice, onset and sign.
- Draws on the global stream would shift the env's command and push sequence between arms and runs (precedent:
  `bc_generator`, ppo_with_extractor.py:638).
- `init_at_random_ep_len` (OPR:277-280) also draws globally. Whether to use it is **OPEN-6**. If yes, use a
  private generator.

### 3.7 Checkpoint format (loadable by `_load_extra` and P2L:920-925)

```text
torch.save({"model_state_dict": student.state_dict(),            # every S1 key incl. critic.*, std_raw; no '_orig_mod.' / 'module.' prefixes
            "estimator_state_dict": ck["estimator_state_dict"],  # S1's, VERBATIM (the launcher's estimator identity refuses otherwise)
            "iter": it, "infos": {provenance: S1 sha, arm, mask, seed, sn_dagger sha, sn_common sha, RESTANCE_D3, beta/sigma spec},
            "optimizer_state_dict": opt.state_dict()},           # optional for the launcher, needed by play.py / evaluation.py
           path)                                                 # omit alg_state (rebuilt from iter, OPR:806-818)
```

- S1's file holds `cuda:0` storages. A checkpoint saved from GPU tensors loads only with CUDA unless
  `map_location` is given, and `runner.load` passes none (OPR:762).
- Saving CPU copies (`{k: v.detach().cpu()}`) lets the CPU round-trip test use the real load path. The values are
  bitwise the same after `load_state_dict`.
- **Round-trip test (CPU):** build a fresh `CrabHexActorCriticRMA` and `DefaultEstimator` from a deepcopy of the
  S1 agent cfg. Load strictly. Check the key set {model_state_dict, estimator_state_dict, iter, infos} (OPR:761-804),
  and that `act_inference(x, hist_encoding=True)` is bitwise equal to the trainer's student on a fixed batch.
- Save every 500 iterations (step 7). The final checkpoint is the judged one.

### 3.8 Trainer unit tests (CPU)

1. R labels equal `RST.restance_target` with the D3 params, and `format_restance` round-trips.
2. Mask parity with the launcher (§2.7 item 3).
3. The routing function is pure and correct on stub tensors: stand rows take `a_stand`, others take `a_s1`; the
   obs is not written; noise appears only on stand rows.
4. β, σ and control draws come from private generators, and `torch.get_rng_state()` is unchanged.
5. Checkpoint round-trip (§3.7).
6. Reservoir cap, LIFT-onset detection, and the stand-end read on the last stand row (never a post-reset row).

---

## 4. A1 `recover-probe` mode (same trainer, `--mode recover_probe`)

- **Configuration:** the same env, routing and logging as §3, with: no student forward, no optimizer step, no
  checkpoint; σ = 0 throughout; β = 1 on every stand. Pushes and domain randomization stay on.
  - Assert in the smoke: zero optimizer steps; executed stand rows are R or spoiler only.
- **Per-stand assignment at entry** (private generator): control (R from entry) with probability ½; otherwise
  statue, chatter or stuck lift with ⅙ each. Spoiler window ~ U[1, 4] s (50–200 steps at `step_dt` 0.02), then R
  for the rest of the stand.
- **Spoiler actions.** Legs are FL, FR, ML, MR, RL, RR = 0..5. Leg i owns action columns cam i, hip 6+i and knee
  12+i (RST:94-97). `SIDE` = (+1, −1, +1, −1, +1, −1) (RST:90). Cam of leg i is at obs col 18+2i.
  - **statue:** `a = 0` (18). Smoke: executed rows are exactly 0, and the ±1 clamp keeps them at 0.
  - **chatter:** `a = R_target + 0.3·randn(18)` (private generator). The wrapper's clamp lowers the realised σ
    where `R_target` is near ±1. Report σ before and after the clamp.
  - **stuck lift** on one random leg l with sign s = ±1 drawn per stand: `a = R_target.clone()`;
    `a[6+l] = lift = −0.5`; `a[12+l] = −rho·SIDE[l]·(−0.5)` (R's knee coupling, RST:275-276);
    `a[l] = clamp(−k_c·(c_l − s·0.4), −1, 1)` with k_c = 2. This is R's own in-air cam law (RST:269) shifted to
    target s·0.4 rad.
    - The cam channel is a **velocity** (π rad/s per unit, slew-limited) with a one-step action delay. ±0.4 rad is
      reached only through this closed loop.
    - Smoke: on most stuck-lift stands at takeover, the held leg is up (obs col 29+l ≤ p_up = −0.0625 rad) and
      |c_l| > 0.3.
- **Scoring:**
  - A stand is scored when its scheduled length is ≥ 8 s and it is not cut by the episode timeout. The definition
    is OPEN-7 when stand → stand resamples merge.
  - **Review 2026-10-12 (amendment to OPEN-7, to log in CHARTER §3 before the A1 read):** the length, the fall and
    the |c| read are the stand's FIRST draw (the entry draw's `time_left`, never merge-extended; a fall before that
    draw's scheduled end; |c| on its last stand row). Only stands that survived their first draw get merged, so a
    floor on the merged length made inclusion depend on the outcome. Stands in progress at the run's first step
    (`entry_kind` run_start) are excluded. Records keep the merged fields as reported only (`sn_dagger_spec.scored`).
  - Fall → fall=1, homed=0. Falls before takeover are reported separately.
  - |c| is read on the last stand row before the exit.
  - Also report homed by R-time bin, and the share of loaded legs with |c| ∈ (0.1, 0.3].
- **Rule** (fixed before the read, PLAN A1): PASS if, for every spoiler, homed ≥ 0.8 × control and stand-fall
  share ≤ control + 0.01. Otherwise STOP and ask.

---

## 5. A2 `A-branch`: what its loader needs (only if the step-9 trigger holds)

1. **The trunk loads unchanged through `_load_extra`** if `model_state_dict` stays S1-shaped, including the unused
   flat head `actor.actor_backbone.6.*`.
   - `runner.load` ignores unknown top-level keys (OPR:761-804), so put the classifier and branch heads in one
     extra key, e.g. `branch_head_state_dict`.
   - Extra keys **inside** `model_state_dict` break the strict load (ACE:261-263).
   - Extra policy-cfg keys raise at construction (ACE:153-158).
2. **Load the head yourself**, from the same file. Build the head module **inside** p2r's RNG save/restore window:
   wrap `p2r._load_extra` so the head is built in that call, which runs inside P2L:949-961. Otherwise `nn.Linear`
   init consumes the global RNG and breaks the identity property (`rng_restored` raises at P2L:962-964).
3. **Forward:** `act_inference` exposes only the final 18-d output. Branch heads need the 128-d penultimate
   features, `actor_backbone[:-1](cat(prop, scan_latent, est_priv, hist_latent))`. Recompose them from the Actor
   internals, mirroring ACE:97-112 op for op: `scan_encoder` on 75:207, `infer_hist_latent` on the last 750 cols,
   with the estimate in 207:216 and the mask after it, via the same `sn_common.stand_net_input`. Per leg, take the
   argmax branch; leg i's action columns are cam i, hip 6+i and knee 12+i.
4. **Run_meta conflict to resolve before building:** PLAN A2 names the variant `stand_net_branch` and requires
   SNC's head-validity rule. But `SNC.stand_net_reasons` requires `variant == "stand_net"` exactly (SNC:177-178).
   The options are (a) write variant `stand_net` plus `branch_head: true`, or (b) amend SNC campaign-locally and
   log the change in CHARTER §3 before any read. `specialist.sha256` hashes the whole file, heads included.

---

## 6. Head-validity `run_meta` contract (what SNC checks; fail closed)

Per stamp (`stamp_reasons` kind `stand_net`, SNC:226-258):

- **`obs0_reasons`** (SNC:139-152): `env_vars.KRABBY_OMNI_STAND_OBS` exactly `"0"`; absent fails.
  `KRABBY_OMNI_STAND_PROXY` and `KRABBY_OMNI_STAND_MASK` are off.
- **`stand_net_reasons`** (SNC:169-189):
  - `p2r_mode` is present;
  - `variant == "stand_net"`;
  - `s1_checkpoint_sha256 ==` REFS suite@1 `checkpoint_sha256` (`968cc550…`);
  - `specialist.sha256` is 64-char lowercase hex;
  - `stand_mask` is int 0/1 or string `"0"`/`"1"`. **A bool is malformed.**
- **`p2_read.invalid_reasons`** (p2_read.py:586-602): ENV4 (p2_read.py:275-276: CLOCK_HIDE `"1"`, STD_PARAM `soft`,
  SPEED_ELLIPSE `0.35:0.35:0.25`, FACING_OBS_CLIP `0.6`); the row env block equals REFS's; the code-state sidecar
  equals REFS's or is bridged. Plus `walkstop_flat_fails` (p2_read.py:1895-1908).
- **Across a head** (`stand_net_consistency`, SNC:192-206): one stand-ckpt sha and one mask per head. A-mask and
  A-full must therefore use different head names.
- **Launcher-side guarantees:** the S1 sha equals `run_meta.checkpoint_sha256` (P2L:1087-1088), and a stamp is
  refused when the wrapper never acted (P2L:1080-1082). The G3 exit clause needs `p2r_stand_mode` in every raw
  episode.

---

## 7. Risks: silent failure modes

**Launcher**

- **L1:** Setting `variant="stand_net"` without overriding `__call__` falls into the probe branch (P2L:599-607). S1
  acts on every row and the stamp looks like a valid stand_net run that is really REFS.
- **L2:** Masking the harness obs in place (`stand_mask_obs(obs, inplace=True)`, or the env-level
  `KRABBY_OMNI_STAND_MASK`) makes S1 see masked frames, which violates D1. It also fails `obs0_reasons`, but only
  after the GPU run.
- **L3:** Masking before the estimate changes the estimate, because the estimator reads cols 65-74.
- **L4:** `stand_mask` written as a bool makes every stamp of the head INVALID (SNC:164).
- **L5:** The stand checkpoint's own estimator is what runs (P2L:595). A trainer that saved a modified estimator
  would change the stand net's cols 207:216 without notice. The §2.3-4 identity refusal closes this.
- **L6:** Importing torch, or loading R by path, at the top of the wrapper or `sn_common` happens before Kit, which
  changes process state. The identity bridge could then miss for a non-code reason.
- **L7:** Patching a module instance that other importers share leaks the patches into them. Use a private
  by-path instance.
- **L8:** `launcher_sha256` names only the phase-2 file (P2L:1045). Without `stand_net.wrapper_sha256` and
  `common_sha256`, a changed wrapper leaves no trace in run_meta.
- **L9:** `EVAL_EXTRA_ARGS` is unquoted (RUN_ARM:532), so a path with spaces or glob characters splits or expands
  silently. `lr_eval` passes no extra args (RUN_ARM:552-553), so env fallbacks are needed. Inherited `P2R_*`
  variables reach P2L's fallbacks; `P2R_BANK_MAX` is accepted silently.
- **L10:** An R or Z env stamp (STAND_OBS=clock), or STAND_OBS set in `P2_ENV_EXTRA`, breaks D5 obs parity. The
  wrapper's pre-Kit refusal catches it; without that refusal it would show up only at read time.
- **L11:** Calling the stand policy without `hist_encoding=True` uses the priv-latent path (ACE:253 default).

**Trainer**

- **T1:** A forward path that differs from the launcher's trains a different function, and open-loop loss will not
  show it. Examples: true priv cols instead of the estimate; `hist_encoding=False`; mask before estimate; a
  different mask row set on stand rows.
- **T2:** Overlay keys missing:
  - `KRABBY_RSI_FRAC` absent → preset 0.2 applies, so RSI rows are staged as SLOT_MOVE (CMD:597-605) and the RSI
    bank is used.
  - `KRABBY_OMNI_SPEED_ELLIPSE` absent → preset 0.35:0.25:0.25.
  - `KRABBY_OMNI_CLOCK_HIDE` absent → the clock is visible in cols 15-16.
  - `KRABBY_PHASEOUT` absent → preset ramps.
  - `KRABBY_STD_PARAM` / `KRABBY_INIT_STD` absent → the strict load fails, which is loud.
  - The values were checked against `phase_env("2a")`.
- **T3:** Dropping the pool and raising only SLOTS falls back to the legacy 4-slot sampler: different walking
  directions, and no failure. A pool and SLOTS that do not match is loud (CMD:313-320).
- **T4:** `RestanceParams()` or `"default"` labels with a different R (§3.5).
- **T5:** Trainer draws on the global RNG shift the command and push streams between arms and runs (§3.6).
- **T6:** An extra `env.get_observations()` mid-run (or a runner built on the real env, OPR:120) pushes an extra
  history frame (OBS:248-252). History then differs from deployment and nothing flags it.
- **T7:** Stepping `env.unwrapped`, or using a wrapper without `clip_actions`, executes rod actions in (1, 4.8]
  (crab_hex_env_cfg.py:272), which deployment never produces.
- **T8:** Fitting the clipped action, or reporting the unclipped mean as "executed", changes what is learned or
  measured (§3.5).
- **T9:** Exit and homed reads on a post-reset row: the obs `env.step` returns for a done row belongs to the new
  episode and slot.
- **T10:** The scan encoder is trainable by default. Its input includes current-step last-action and proxy cols
  75-88, the copycat channel A-mask removes, and the true priv_explicit 197-205. Freezing it or not is a design
  choice (OPEN-1).
- **T11:** An optimizer built over trainable params only, or a checkpoint with no `optimizer_state_dict`, loads in
  the launcher but fails in play.py and evaluation.py (`load_optimizer=True`).
- **T12:** Setting train.py's TF32 flags (train.py:109-112), or computing the student only on the stand-row
  subset, can break step-6 bitwise action parity on GPU even when the math is the same.
- **T13:** The stand-heavy pool (about 6 % non-voting episodes, about 4.8 m commanded distance per episode against
  about 8.1 m for S1, from the env reader's simplified Monte Carlo) and the 3–12 s resample are a regime S1 was not
  trained in. Exits feed promotion (`ep_track_dist`, omni_parkour_event.py:352-358). R slices and student slices
  share terrain levels only if they share envs and iterations.
- **T14:** RUN_ARM's `drivers_idle` does not see `sn_dagger.py`, so a row eval could start concurrently. The
  campaign queue must serialize them (one Isaac process at a time; memory: GPU queue chaining).

---

## 8. Open unknowns and decisions left to builders

**Unknowns (not determinable by reading)**

- **U1:** Whether the identity bridge is **bitwise** equal to REFS on GPU. P2R_PROBE_s18 proved extra-runner
  construction and RNG restore, but not a second forward replacing S1's actions. The rule to apply on a miss is
  still to be pre-stated (PLAN header).
- **U2:** Bitwise trainer↔launcher action parity across processes and batch shapes: eval batches are 100 or more
  rows; the trainer's is 256. Within one process on an identical batch it is testable offline. Otherwise state an
  atol (PLAN "Left for implementation").
- **U3:** GPU time per iteration and buffer growth. These are measured in the step-6 smoke, which sets the
  iteration count for the 2-GPU-h cap.
- **U4:** How S1 walks, tracks and gets promoted under 3–12 s resampling with a 0.5 stand share. No stamp covers
  that regime.

**Decisions (OPEN; record each in CHARTER §3)**

- **OPEN-1:** Whether `student.actor.scan_encoder` trains (PLAN header item).
- **OPEN-2:** Learning rate, gradient steps per iteration, env steps per iteration, reservoir cap and minibatch size.
- **OPEN-3:** Branch balancing at row level vs leg level. One option is per-leg loss weights on that leg's three
  action columns by inverse branch frequency. Another is a row key, e.g. the highest-priority branch across legs.
- **OPEN-4:** The 10 % control slice as a fixed env subset or a per-episode draw. A fixed subset couples terrain
  level trajectories to the executor. A per-episode draw keeps envs exchangeable.
- **OPEN-5:** DART σ draw probabilities over {0, 0.1, 0.3}. Also whether σ applies to R-executed β episodes.
- **OPEN-6:** `init_at_random_ep_len`, which PLAN does not cover. PPO used it (train.py:197).
- **OPEN-7:** Defining a "stand" and its "scheduled length" when stand → stand resamples merge into one latched
  stand. Only the first draw's `time_left` is known at entry (command_manager.py:121-150, 152-167, 184: a reset entry has
  `time_left = draw − dt`; a mid-episode entry has `time_left = draw`).
- **OPEN-8:** The A2 variant name vs SNC (§5.4).
- **OPEN-9:** Whether stand_net records the R raws now (§2.4 recommendation) or adds them later for the observe pass.

---

## 9. Reader conflicts and how they were resolved (by re-reading the source)

- **C1, launcher mechanism.** One reader proposed wrapping `st_policy` in a mask lambda, keeping variant `st2` and
  relabelling in `p2r_block`. Another proposed a subclass with a full `__call__` override.
  **Chosen: the subclass.** It writes the swap-live raws (P2L:507-511), and it puts the label on the mode object, so
  the write-time print (P2L:1122) and `p2r_block` agree without relabelling. The lambda route remains a valid
  minimal fallback, but it produces no `sn_*` raws.
- **C2, mask rows.** `rows=stand` and `rows=None` give identical stand-row inputs: masked values are constants
  (RST:330-338), rows are independent, and `torch.where` discards the rest. **Chosen: `rows=None`**, so the shared
  function takes no stand argument.
- **C3, building S1 and the student in the trainer.** Direct construction avoids OPR:120's history push and PPO
  extras. The factory on a StandInEnv is the alternative (§3.3).
- **C4, "unknown" items another reader resolved.** All re-read here:
  - env step order (ENV:95-165);
  - obs aliasing: a fresh tensor per compute (observation_manager.py:395, :433);
  - S1 uses the scan encoder (ACE:31, S1-agent:74);
  - `crab_failure` thresholds in S1's training env: 0.5 rad and 500 N (S1-env:2340-2342);
  - action delay is 1 throughout (S1-env:796-806).
- **C5, eval task.** The REFS eval task is `Isaac-Crab-Hex-Omni-Walk-Play-v0`
  (`CrabHexOmniFlatWalkEnvCfgPLAY`, from REFS `run_meta.task`). The trainer uses
  `Isaac-Crab-Hex-Omni-Teacher-v0` per D2, as S1's own training did. Both declare the same observation cfg class,
  `CrabHexOmniObservationsCfg` (crab_hex_omni_env_cfg.py:279, :321). Their termination and action cfgs differ
  (:280-283 vs :322-325), so step-6 D5 parity must compare obs columns, not cfg dumps.
- **C6, stale docstring.** The p2_read docstring's claim that ST2M's top-level `checkpoint_sha256` is ST2's
  (p2_read.py:2657-2663) is stale: the ST2M stamp records S1's (launcher reader, from that stamp's run_meta). No
  builder action is needed beyond not relying on that docstring.
