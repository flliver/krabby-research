```
=== crab-hex gait eval ===
scenario   : teacher_2b2_forward  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/gait_tuned/t2_hybrid/logs/rsl_rl/crab_hex_teacher/2026-08-09_23-17-24/model_20696.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2717524523238392  p25=0.15593906198884508  p75=0.3405979730315397
tippy_tap_fraction median=0.21350924870693433
slip_ratio         median=0.045772603145587204
schedule_completion_rate=0.6
terminations={'schedule_complete': 6, 'fall': 4}

by hold:
       low: tripod median=0.43066996472843944 (n=7)
       mid: tripod median=0.21759472740549077 (n=8)
      high: tripod median=0.19978609454723223 (n=6)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
