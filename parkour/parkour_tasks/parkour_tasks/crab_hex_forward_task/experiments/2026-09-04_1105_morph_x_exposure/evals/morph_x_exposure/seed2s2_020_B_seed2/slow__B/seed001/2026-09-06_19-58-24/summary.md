```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_13-42-22/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5778707855432069  p25=0.5276073986897553  p75=0.6307927605050176
tippy_tap_fraction median=0.21403940886699507
slip_ratio         median=0.2087789635616178
tracking_ratio     median=0.6498579738180935  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.96
terminations={'schedule_complete': 96, 'fall': 4}

by hold:
     stand: cmd 0.00 -> achieved 0.077 m/s | tripod median=0.47282745778929236 (n=100)
     creep: cmd 0.25 -> achieved 0.181 m/s | tripod median=0.640620391875614 (n=97)
       low: cmd 0.35 -> achieved 0.203 m/s | tripod median=0.6701610789553443 (n=96)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
