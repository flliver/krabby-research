```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_05-18-20/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5523278747180345  p25=0.5090015368538628  p75=0.579451541407236
tippy_tap_fraction median=0.25992793333833797
slip_ratio         median=0.23807797790454538
tracking_ratio     median=0.4470879650911343  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.89
terminations={'fall': 11, 'schedule_complete': 89}

by hold:
     stand: cmd 0.00 -> achieved 0.069 m/s | tripod median=0.6353158911572463 (n=100)
     creep: cmd 0.25 -> achieved 0.119 m/s | tripod median=0.5458617996116343 (n=97)
       low: cmd 0.35 -> achieved 0.146 m/s | tripod median=0.4872391404288774 (n=90)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
