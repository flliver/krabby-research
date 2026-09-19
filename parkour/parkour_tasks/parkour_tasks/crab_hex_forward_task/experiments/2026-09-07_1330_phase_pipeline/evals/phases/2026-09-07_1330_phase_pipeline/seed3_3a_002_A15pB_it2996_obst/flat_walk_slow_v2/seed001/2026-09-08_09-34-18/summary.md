```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-07_23-13-07/model_24995.pt
episodes   : 100  unscored: 95
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.030179543282758237  p25=0.019172318033221652  p75=0.0438300914415541
tippy_tap_fraction median=0.5384615384615384
slip_ratio         median=0.2831463947423599
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.197 m/s | tripod median=0.030179543282758237 (n=5)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
