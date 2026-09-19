```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_13-42-22/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.43879513104009826  p25=0.38611114901695354  p75=0.5116073090473945
tippy_tap_fraction median=0.22257607926397735
slip_ratio         median=0.20876015509779933
tracking_ratio     median=0.5597280686185051  (achieved/commanded vx, walking holds; n=88)
schedule_completion_rate=0.72
terminations={'schedule_complete': 72, 'fall': 28}

by hold:
     stand: cmd 0.00 -> achieved 0.060 m/s | tripod median=0.4096856538440883 (n=100)
     creep: cmd 0.25 -> achieved 0.161 m/s | tripod median=0.466794050971422 (n=88)
       low: cmd 0.35 -> achieved 0.162 m/s | tripod median=0.49412921782520614 (n=75)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
