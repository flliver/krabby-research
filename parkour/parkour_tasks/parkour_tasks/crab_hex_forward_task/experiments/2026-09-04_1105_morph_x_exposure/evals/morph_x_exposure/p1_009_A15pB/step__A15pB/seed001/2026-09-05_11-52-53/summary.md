```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_05-27-19/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5011681636346232  p25=0.4384905327938734  p75=0.553279073658067
tippy_tap_fraction median=0.29434564523483814
slip_ratio         median=0.3075308067302468
tracking_ratio     median=0.6701543665697263  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}

by hold:
     stand: cmd 0.00 -> achieved 0.025 m/s | tripod median=0.43965772030326644 (n=100)
     creep: cmd 0.25 -> achieved 0.204 m/s | tripod median=0.5370066406893954 (n=100)
       low: cmd 0.35 -> achieved 0.186 m/s | tripod median=0.5467243600631451 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
