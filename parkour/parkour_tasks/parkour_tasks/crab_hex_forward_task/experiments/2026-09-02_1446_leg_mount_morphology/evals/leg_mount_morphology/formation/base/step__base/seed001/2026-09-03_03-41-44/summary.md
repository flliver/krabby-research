```
=== crab-hex gait eval ===
scenario   : step__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_21-12-20/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4627894197243735  p25=0.4160523858206997  p75=0.5309551495454176
tippy_tap_fraction median=0.27173486409577075
slip_ratio         median=0.2767561935191819
tracking_ratio     median=0.4381038092422965  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.31
terminations={'fall': 69, 'schedule_complete': 31}

by hold:
     stand: cmd 0.00 -> achieved 0.033 m/s | tripod median=0.40743724568205997 (n=100)
     creep: cmd 0.25 -> achieved 0.111 m/s | tripod median=0.5615029519024431 (n=99)
       low: cmd 0.35 -> achieved 0.136 m/s | tripod median=0.37621039324596073 (n=67)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
