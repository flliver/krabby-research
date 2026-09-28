```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_15-49-15/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.30132557133641125  p25=0.23259842498419053  p75=0.3458469379394234
tippy_tap_fraction median=0.354111084162018
slip_ratio         median=0.5190389171842171
tracking_ratio     median=0.639653744728376  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.54
terminations={'schedule_complete': 54, 'fall': 46}

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.0 (n=99)
     creep: cmd 0.25 -> achieved 0.194 m/s | tripod median=0.5313874304181536 (n=97)
       low: cmd 0.35 -> achieved 0.172 m/s | tripod median=0.4093723474364664 (n=72)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
