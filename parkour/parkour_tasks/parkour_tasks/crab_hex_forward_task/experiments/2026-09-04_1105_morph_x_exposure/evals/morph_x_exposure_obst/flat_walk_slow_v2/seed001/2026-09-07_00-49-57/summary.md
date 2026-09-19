```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_18-32-32/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5776010827003184  p25=0.5147843245151911  p75=0.6230146039490135
tippy_tap_fraction median=0.2138364779874214
slip_ratio         median=0.16665596703590085
tracking_ratio     median=0.6785313905318642  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}

by hold:
     stand: cmd 0.00 -> achieved 0.073 m/s | tripod median=0.5780810220010597 (n=100)
     creep: cmd 0.25 -> achieved 0.188 m/s | tripod median=0.5567830155046386 (n=99)
       low: cmd 0.35 -> achieved 0.217 m/s | tripod median=0.6143096669275563 (n=97)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
