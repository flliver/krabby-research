```
=== crab-hex gait eval ===
scenario   : creep20__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-09-02_21-12-20/model_4999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.621834245908466  p25=0.5505495252318479  p75=0.6677554903896119
tippy_tap_fraction median=0.30810147299509005
slip_ratio         median=0.2551792410748819
tracking_ratio     median=0.46375987576091576  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}

by hold:
   creep_a: cmd 0.25 -> achieved 0.113 m/s | tripod median=0.6362429551099988 (n=95)
   creep_b: cmd 0.25 -> achieved 0.116 m/s | tripod median=0.6139075733622518 (n=90)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
