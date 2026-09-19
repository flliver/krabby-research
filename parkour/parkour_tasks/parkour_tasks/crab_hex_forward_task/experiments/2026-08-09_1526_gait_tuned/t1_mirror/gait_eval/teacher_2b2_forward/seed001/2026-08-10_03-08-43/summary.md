```
=== crab-hex gait eval ===
scenario   : teacher_2b2_forward  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/gait_tuned/t1_mirror/logs/rsl_rl/crab_hex_teacher/2026-08-09_22-47-51/model_20696.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.32076806996530655  p25=0.09347906602563923  p75=0.3761631906062664
tippy_tap_fraction median=0.2713178294573643
slip_ratio         median=0.05152805897207638
schedule_completion_rate=0.7
terminations={'fall': 3, 'schedule_complete': 7}

by hold:
       low: tripod median=0.3556739312056945 (n=9)
       mid: tripod median=0.360625234476713 (n=7)
      high: tripod median=0.2882114739940419 (n=7)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
