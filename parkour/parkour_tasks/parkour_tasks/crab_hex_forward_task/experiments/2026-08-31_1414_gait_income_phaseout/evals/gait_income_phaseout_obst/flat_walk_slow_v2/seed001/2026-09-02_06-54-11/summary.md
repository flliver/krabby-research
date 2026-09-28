```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6039209827229536  p25=0.5526149424081208  p75=0.6489761282050134
tippy_tap_fraction median=0.23858974358974358
slip_ratio         median=0.23556328952588412
tracking_ratio     median=0.6251558720212815  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.27
terminations={'fall': 73, 'schedule_complete': 27}

by hold:
     stand: cmd 0.00 -> achieved 0.069 m/s | tripod median=0.7188463178077864 (n=100)
     creep: cmd 0.25 -> achieved 0.156 m/s | tripod median=0.48376825449806427 (n=97)
       low: cmd 0.35 -> achieved 0.130 m/s | tripod median=0.43305020557104457 (n=40)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
