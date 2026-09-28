```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_15-49-15/model_4999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.37703982861592894  p25=0.3262873074872389  p75=0.4237464482713448
tippy_tap_fraction median=0.33821025750159606
slip_ratio         median=0.46196623521770863
tracking_ratio     median=0.6850173982868746  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.82
terminations={'schedule_complete': 82, 'fall': 18}

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.0 (n=99)
     creep: cmd 0.25 -> achieved 0.204 m/s | tripod median=0.5778472297768822 (n=93)
       low: cmd 0.35 -> achieved 0.192 m/s | tripod median=0.5894470228756648 (n=82)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
