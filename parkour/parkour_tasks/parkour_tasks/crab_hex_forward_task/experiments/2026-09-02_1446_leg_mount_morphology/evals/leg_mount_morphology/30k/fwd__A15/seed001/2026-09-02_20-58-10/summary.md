```
=== crab-hex gait eval ===
scenario   : fwd__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5153072208705477  p25=0.4880994221522452  p75=0.5397174987302888
tippy_tap_fraction median=0.30951621477937263
slip_ratio         median=0.2849056772328431
tracking_ratio     median=0.2796528025768349  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.72
terminations={'schedule_complete': 72, 'fall': 28}

by hold:
       low: cmd 0.30 -> achieved 0.115 m/s | tripod median=0.5845153992774794 (n=100)
       mid: cmd 0.47 -> achieved 0.122 m/s | tripod median=0.5199820685536871 (n=100)
      high: cmd 0.65 -> achieved 0.131 m/s | tripod median=0.4308504978650658 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
