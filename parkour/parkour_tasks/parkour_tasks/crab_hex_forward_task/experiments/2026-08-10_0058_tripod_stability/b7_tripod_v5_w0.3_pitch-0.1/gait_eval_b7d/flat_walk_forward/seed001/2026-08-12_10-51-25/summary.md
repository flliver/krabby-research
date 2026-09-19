```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b7_tripod_v5_w0.3_pitch-0.1/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_05-56-51/model_27996.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3829679221011919  p25=0.37225910664568146  p75=0.3961805283555505
tippy_tap_fraction median=0.07051118821020466
slip_ratio         median=0.02378439464458988
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.48766471572660886 (n=10)
       mid: tripod median=0.3966328535262936 (n=10)
      high: tripod median=0.25189759802204365 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
