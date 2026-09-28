```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_00-09-00/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4106399340268905  p25=0.3544926972733591  p75=0.46278950307259115
tippy_tap_fraction median=0.33016584491259526
slip_ratio         median=0.5361807719462937
tracking_ratio     median=0.6033296320840429  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.86
terminations={'schedule_complete': 86, 'fall': 14}

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.0 (n=94)
     creep: cmd 0.25 -> achieved 0.175 m/s | tripod median=0.586864948096832 (n=99)
       low: cmd 0.35 -> achieved 0.171 m/s | tripod median=0.6596782288100231 (n=88)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
