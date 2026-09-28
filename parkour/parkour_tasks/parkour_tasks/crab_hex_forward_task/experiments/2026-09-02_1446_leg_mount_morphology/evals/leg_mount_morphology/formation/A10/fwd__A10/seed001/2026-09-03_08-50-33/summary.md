```
=== crab-hex gait eval ===
scenario   : fwd__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_02-14-33/model_4999.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4622793901426851  p25=0.40095624093352483  p75=0.5019158301609988
tippy_tap_fraction median=0.2765912185159973
slip_ratio         median=0.2095611040398264
tracking_ratio     median=0.5852444021048588  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
       low: cmd 0.30 -> achieved 0.230 m/s | tripod median=0.43166230323435284 (n=97)
       mid: cmd 0.47 -> achieved 0.192 m/s | tripod median=0.4970383008862116 (n=86)
      high: cmd 0.65 -> achieved 0.156 m/s | tripod median=0.4344713412352271 (n=7)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
