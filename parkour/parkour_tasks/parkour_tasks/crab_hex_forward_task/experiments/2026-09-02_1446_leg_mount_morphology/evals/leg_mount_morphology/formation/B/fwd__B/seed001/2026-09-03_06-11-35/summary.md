```
=== crab-hex gait eval ===
scenario   : fwd__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_23-43-36/model_4999.pt
episodes   : 100  unscored: 8
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.47205528433713406  p25=0.4226308441882644  p75=0.518373399682574
tippy_tap_fraction median=0.36666666666666664
slip_ratio         median=0.2998731930845481
tracking_ratio     median=0.4327250639542592  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
       low: cmd 0.30 -> achieved 0.160 m/s | tripod median=0.5584323823540182 (n=92)
       mid: cmd 0.47 -> achieved 0.157 m/s | tripod median=0.2738515320861928 (n=75)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
