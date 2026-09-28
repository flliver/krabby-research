```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_23-43-36/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.24669442171728034  p25=0.21325789300978065  p75=0.2924115484103951
tippy_tap_fraction median=0.34991304347826085
slip_ratio         median=0.5349834194477752
tracking_ratio     median=0.5242332726432453  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.31
terminations={'fall': 69, 'schedule_complete': 31}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.02051458674602974 (n=98)
     creep: cmd 0.25 -> achieved 0.148 m/s | tripod median=0.4261931420856333 (n=100)
       low: cmd 0.35 -> achieved 0.111 m/s | tripod median=0.3310501052372621 (n=59)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
