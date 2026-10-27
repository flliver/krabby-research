```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_flat_walk/2026-09-11_10-57-55/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.27865082216432524  p25=0.2222231478903568  p75=0.34953134117411894
tippy_tap_fraction median=0.4112394957983193
slip_ratio         median=0.48863190565498643
tracking_ratio     median=0.4952117343371064  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.77
terminations={'fall': 23, 'schedule_complete': 77}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.006841313066798545 (n=98)
     creep: cmd 0.25 -> achieved 0.150 m/s | tripod median=0.44550441804344204 (n=99)
       low: cmd 0.35 -> achieved 0.132 m/s | tripod median=0.4158270804068473 (n=88)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
