```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_21-37-35/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3344177826987544  p25=0.2683542543312344  p75=0.3900014931564092
tippy_tap_fraction median=0.3737697085523173
slip_ratio         median=0.5736091578426866
tracking_ratio     median=0.5684143255502778  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.68
terminations={'schedule_complete': 68, 'fall': 32}

by hold:
     stand: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.0 (n=99)
     creep: cmd 0.25 -> achieved 0.164 m/s | tripod median=0.4805728380055527 (n=99)
       low: cmd 0.35 -> achieved 0.161 m/s | tripod median=0.5621651993549287 (n=80)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
