```
=== crab-hex gait eval ===
scenario   : fwd__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.48388136004058263  p25=0.4632366973779302  p75=0.49890337179363464
tippy_tap_fraction median=0.28439594027979154
slip_ratio         median=0.2678611579119956
tracking_ratio     median=0.3109567293720416  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.94
terminations={'schedule_complete': 94, 'fall': 6}

by hold:
       low: cmd 0.30 -> achieved 0.133 m/s | tripod median=0.5918737171364232 (n=100)
       mid: cmd 0.47 -> achieved 0.129 m/s | tripod median=0.45942233800098353 (n=100)
      high: cmd 0.65 -> achieved 0.146 m/s | tripod median=0.38396202272054586 (n=97)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
