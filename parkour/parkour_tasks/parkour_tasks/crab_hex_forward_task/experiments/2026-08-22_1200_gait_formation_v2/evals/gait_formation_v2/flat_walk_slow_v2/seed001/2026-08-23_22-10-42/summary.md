```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_16-18-56/model_14997.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5668372086350724  p25=0.5367240211862412  p75=0.5892132957140318
tippy_tap_fraction median=0.22151898734177214
slip_ratio         median=0.23164844203015011
tracking_ratio     median=0.4724297733207831  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.024 m/s | tripod median=0.510120491673965 (n=96)
     creep: cmd 0.25 -> achieved 0.142 m/s | tripod median=0.6444140431829153 (n=93)
       low: cmd 0.35 -> achieved 0.131 m/s | tripod median=0.5497756681873858 (n=91)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
