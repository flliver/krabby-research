```
=== crab-hex gait eval ===
scenario   : fwd__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5735543136514343  p25=0.5441112775681374  p75=0.5929016139726069
tippy_tap_fraction median=0.28662420382165604
slip_ratio         median=0.2545379364278706
tracking_ratio     median=0.3181321452977039  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
       low: cmd 0.30 -> achieved 0.121 m/s | tripod median=0.6336538443013127 (n=100)
       mid: cmd 0.47 -> achieved 0.142 m/s | tripod median=0.581085691675546 (n=100)
      high: cmd 0.65 -> achieved 0.160 m/s | tripod median=0.5047531087070073 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
