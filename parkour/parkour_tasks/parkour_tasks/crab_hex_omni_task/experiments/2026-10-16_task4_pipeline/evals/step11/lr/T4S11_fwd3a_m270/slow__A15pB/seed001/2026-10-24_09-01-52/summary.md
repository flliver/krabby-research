```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3397496995505662  p25=0.3054827759702124  p75=0.37581678796571455
tippy_tap_fraction median=0.4100462379150904
slip_ratio         median=0.4047998314106935
tracking_ratio     median=0.7335099256900068  (achieved/commanded vx, walking holds; n=58)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.136 m/s | tripod median=0.3597308896815437 (n=97)
     creep: cmd 0.25 -> achieved 0.183 m/s | tripod median=0.23733564355848263 (n=54)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
