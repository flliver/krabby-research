```
=== crab-hex gait eval ===
scenario   : fwd__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5388384045941788  p25=0.5103188261812044  p75=0.5592579797220255
tippy_tap_fraction median=0.2958250579278384
slip_ratio         median=0.2904137615927258
tracking_ratio     median=0.28028143191572186  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.78
terminations={'schedule_complete': 78, 'fall': 22}

by hold:
       low: cmd 0.30 -> achieved 0.110 m/s | tripod median=0.5976767864218167 (n=100)
       mid: cmd 0.47 -> achieved 0.126 m/s | tripod median=0.5495331925464566 (n=100)
      high: cmd 0.65 -> achieved 0.129 m/s | tripod median=0.47277235770728304 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
