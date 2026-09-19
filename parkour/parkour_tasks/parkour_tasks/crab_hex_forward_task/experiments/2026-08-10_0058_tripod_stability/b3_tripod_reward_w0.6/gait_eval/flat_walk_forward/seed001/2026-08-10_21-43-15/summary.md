```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b3_tripod_reward_w0.6/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_17-24-17/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3942995372307062  p25=0.38869024371042715  p75=0.4010759237953417
tippy_tap_fraction median=0.07172306280070904
slip_ratio         median=0.021464627576538723
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5053109906078969 (n=10)
       mid: tripod median=0.41265734405462673 (n=10)
      high: tripod median=0.2800929047024856 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
