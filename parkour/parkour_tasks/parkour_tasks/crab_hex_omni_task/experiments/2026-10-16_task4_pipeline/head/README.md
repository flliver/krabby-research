# 2026-10-16_task4_pipeline — checkpoint of record

**Bundled checkpoint:** `head/model_25992.pt` (12.6 MB, sha256 `03b6009a6e952133f6c21c36c8d315b76f945d431615ef6d6798b7f4a5f59fc7`)

- **Source (not tracked):** `parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt`
- **Role / task:** teacher_2a / `Isaac-Crab-Hex-Omni-Teacher-v0`
- **Plant:** A15+B
- **Why this checkpoint:** F19 T* = p3b (the trained payload head with the highest max_stable that passes its paired 0-kg check under thresholds v1.2); max_stable 135 kg at N_p 3000 iterations per stage (sim plant, force side not enforced); the depth student, SN and SH were trained beside it. F19 = deliverable (user STOP-END 2026-10-24); baked as the omni policy of record (de81f98a; policy/manifest.yaml current = f19_8_p3b_payload180); tag omni-sim2real-baseline-2026-10-24 local, the push is the user's.
- **Environment of record:** `KRABBY_ACTION_LPF_HZ=0 KRABBY_ACTION_LPF_OBS=0 KRABBY_APEX_W=1.0 KRABBY_CLOCK_W=1.0 KRABBY_FLAT_TERRAIN_GEOM=ring KRABBY_INIT_STD=0.8 KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_OMNI_ANG_VEL_CLIP=0.2 KRABBY_OMNI_ANG_VEL_MAX=0.3 KRABBY_OMNI_CLOCK_HIDE=1 KRABBY_OMNI_CLOCK_TURN_R=1.2 KRABBY_OMNI_CMD_OBS=0 KRABBY_OMNI_COMMAND_FRAME=world KRABBY_OMNI_FACING_OBS_CLIP=0.6 KRABBY_OMNI_FACING_OBS_ENCODING=raw KRABBY_OMNI_RELAY_HYST=0 KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 KRABBY_OMNI_STAND_HOLD=0 KRABBY_OMNI_STAND_OBS=0 KRABBY_OMNI_STAND_PARK=0 KRABBY_OMNI_YAW_RANGE=-3.14159:3.14159 KRABBY_PLANT=A15+B KRABBY_PLAY_DEBUG_VIS=0 KRABBY_RSI_FRAC=0 KRABBY_STD_PARAM=soft KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.1`

| metric | value |
|---|---|
| max_stable_kg | 135 (first unstable 180; limiting factor budget_bound; sim plant, force side not enforced) |
| zero_kg_paired_check | PASS, max margin use 0.66 (v1.2 track_err; raw back drop 0.068 / 0.071 WATCH) |
| field_hard_completion_es1_es2 | 0.91 / 0.90 |
| side22 | FAST / FAST |
| report | REPORT_baseline_vs_final.md |

**Extra head:** `head/model_4000.pt` (stand_network (SN_T4F19_sn_s19, beside T*), sha256 `96832099e58b75d1cc86f6f9d98327c56b827c2fac1b6c61c96b07f0a8e97b73`) — source `parkour/outputs/omni_plan/stand_network/runs/SN_T4F19_sn_s19/model_4000.pt`

**Extra head:** `head/model_5000.pt` (stand_height (SH_T4F19_sh_s19, beside T*; warm-started from SN_T4F19_sn_s19), sha256 `c12692343fcc31d114ee69b56f59c4879c4969f03414832d4a6a3edbbfe55927`) — source `parkour/outputs/omni_plan/stand_height/runs/SH_T4F19_sh_s19/model_5000.pt`

**Extra head:** `head/student_c28000_policy.onnx` (student c28000 ONNX deploy export -- policy (GRU state explicit; checkpoint sha256 4fa81f0f...), sha256 `4fb3efe11ace47647e20826fc5fca58e5a350ab2a3c05ff8ca5b3e042629d148`) — source `parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/exported_deploy/policy.onnx`

**Extra head:** `head/student_c28000_depth_latest.onnx` (student c28000 ONNX deploy export -- depth encoder (4 cameras), sha256 `2dd76ac94a6cd1ca71209113ece6e6431720cbb26f31953346136239314941bd`) — source `parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/exported_deploy/depth_latest.onnx`

**Extra head:** `head/student_c28000_omni_student_io.json` (student c28000 ONNX deploy export -- I/O contract (obs / depth / GRU shapes), sha256 `4ee03d02cbf1f6736d4fa392d18bc36af0e893e008543e2f744cac65439bff1a`) — source `parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/exported_deploy/omni_student_io.json`

## Play / evaluate

```bash
cd parkour
env $(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /') OMNI_KIT_ACCEPT_EULA=yes KRABBY_ACTION_LPF_HZ=0 KRABBY_ACTION_LPF_OBS=0 KRABBY_APEX_W=1.0 KRABBY_CLOCK_W=1.0 KRABBY_FLAT_TERRAIN_GEOM=ring KRABBY_INIT_STD=0.8 KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_OMNI_ANG_VEL_CLIP=0.2 KRABBY_OMNI_ANG_VEL_MAX=0.3 KRABBY_OMNI_CLOCK_HIDE=1 KRABBY_OMNI_CLOCK_TURN_R=1.2 KRABBY_OMNI_CMD_OBS=0 KRABBY_OMNI_COMMAND_FRAME=world KRABBY_OMNI_FACING_OBS_CLIP=0.6 KRABBY_OMNI_FACING_OBS_ENCODING=raw KRABBY_OMNI_RELAY_HYST=0 KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 KRABBY_OMNI_STAND_HOLD=0 KRABBY_OMNI_STAND_OBS=0 KRABBY_OMNI_STAND_PARK=0 KRABBY_OMNI_YAW_RANGE=-3.14159:3.14159 KRABBY_PLANT=A15+B KRABBY_PLAY_DEBUG_VIS=0 KRABBY_RSI_FRAC=0 KRABBY_STD_PARAM=soft KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.1 /home/nickmagus/krabby/isaac_venv/bin/python scripts/rsl_rl/play.py --task Isaac-Crab-Hex-Omni-Walk-Play-v0 --num_envs 1 --checkpoint ../parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/head/model_25992.pt
```

The `env -u` prefix drops every `KRABBY_*` variable inherited from the shell, so the environment of record above is the whole knob set. play.py writes an export (`exported_teacher/`, JIT + ONNX) beside the checkpoint it plays: copy the head elsewhere first to keep `head/` clean.

The extra heads are not played by this line: see the campaign's own README (and, for a baked head, `crab_hex_omni_task/policy/README.md`) for their play / eval / video commands.

## Paths note

This campaign lives at `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/`. Checkpoint paths inside its records point at the raw artifacts (`parkour/logs/rsl_rl/...` for era-B campaigns, the campaign's own `logs/rsl_rl/...` tree for era A), which stay on disk but are not tracked; the tracked copy of the checkpoint of record is this `head/` directory.

_Generated by `experiments/tools/bundle_experiment.py` on 2026-10-25._
