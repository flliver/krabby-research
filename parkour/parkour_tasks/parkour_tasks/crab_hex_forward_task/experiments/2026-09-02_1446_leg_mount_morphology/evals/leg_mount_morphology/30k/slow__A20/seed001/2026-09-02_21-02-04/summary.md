```
=== crab-hex gait eval ===
scenario   : slow__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6534394346539765  p25=0.6308156795193192  p75=0.6677198544529527
tippy_tap_fraction median=0.248427424140523
slip_ratio         median=0.2344412035318763
tracking_ratio     median=0.3686868716876326  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.060 m/s | tripod median=0.7182725565705816 (n=100)
     creep: cmd 0.25 -> achieved 0.098 m/s | tripod median=0.6188117384776595 (n=100)
       low: cmd 0.35 -> achieved 0.114 m/s | tripod median=0.6249128018746799 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
