```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_21-37-35/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.34005985383248105  p25=0.2848568350306914  p75=0.37879752993365623
tippy_tap_fraction median=0.36791246804030686
slip_ratio         median=0.6364196114064566
tracking_ratio     median=0.5685244097168463  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.67
terminations={'schedule_complete': 67, 'fall': 33}

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.0 (n=98)
     creep: cmd 0.25 -> achieved 0.153 m/s | tripod median=0.4685000962360576 (n=99)
       low: cmd 0.35 -> achieved 0.175 m/s | tripod median=0.6049580603829938 (n=78)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
