```
=== crab-hex gait eval ===
scenario   : fwd__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.45849871017656374  p25=0.432825258201407  p75=0.48449126592232594
tippy_tap_fraction median=0.29934139260799864
slip_ratio         median=0.2908743953572789
tracking_ratio     median=0.30757546446613837  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.6
terminations={'schedule_complete': 60, 'fall': 40}

by hold:
       low: cmd 0.30 -> achieved 0.127 m/s | tripod median=0.5483880715216214 (n=100)
       mid: cmd 0.47 -> achieved 0.129 m/s | tripod median=0.45852891436046916 (n=99)
      high: cmd 0.65 -> achieved 0.143 m/s | tripod median=0.35148185663735965 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
