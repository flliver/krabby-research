```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_05-27-19/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5639496701707825  p25=0.5247696337512733  p75=0.6122564375558739
tippy_tap_fraction median=0.28035113035113035
slip_ratio         median=0.2715400540969287
tracking_ratio     median=0.7526901476966235  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.035 m/s | tripod median=0.33119666318426577 (n=100)
     creep: cmd 0.25 -> achieved 0.217 m/s | tripod median=0.6532727976359121 (n=100)
       low: cmd 0.35 -> achieved 0.226 m/s | tripod median=0.7320267171856297 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
