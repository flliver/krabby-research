```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_19-40-14/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5836056672481188  p25=0.5443924928752468  p75=0.6058999144179249
tippy_tap_fraction median=0.2345679012345679
slip_ratio         median=0.2140589225926499
tracking_ratio     median=0.41549696868018693  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.89
terminations={'schedule_complete': 89, 'fall': 11}

by hold:
     stand: cmd 0.00 -> achieved 0.058 m/s | tripod median=0.6979281884795786 (n=100)
     creep: cmd 0.25 -> achieved 0.110 m/s | tripod median=0.5464816451872965 (n=97)
       low: cmd 0.35 -> achieved 0.138 m/s | tripod median=0.49347331542964057 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
