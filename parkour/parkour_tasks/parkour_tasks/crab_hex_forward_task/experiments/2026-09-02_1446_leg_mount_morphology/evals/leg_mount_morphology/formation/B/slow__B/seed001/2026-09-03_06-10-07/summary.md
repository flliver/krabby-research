```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_23-43-36/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2556732872267352  p25=0.19376961093822923  p75=0.30363468934841037
tippy_tap_fraction median=0.3359498120984362
slip_ratio         median=0.5199329771457302
tracking_ratio     median=0.5344512965052494  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.31
terminations={'schedule_complete': 31, 'fall': 69}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.03471091482727257 (n=96)
     creep: cmd 0.25 -> achieved 0.156 m/s | tripod median=0.4289965541287008 (n=100)
       low: cmd 0.35 -> achieved 0.128 m/s | tripod median=0.3511517962950228 (n=65)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
