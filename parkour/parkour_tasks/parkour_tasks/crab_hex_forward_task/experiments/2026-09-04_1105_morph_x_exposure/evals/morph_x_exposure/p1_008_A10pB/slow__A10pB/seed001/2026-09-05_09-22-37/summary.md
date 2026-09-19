```
=== crab-hex gait eval ===
scenario   : slow__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_03-00-33/model_4999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3992701196735455  p25=0.32780596827266784  p75=0.4582794475156671
tippy_tap_fraction median=0.30921338548457195
slip_ratio         median=0.23008171577065736
tracking_ratio     median=0.6146840943817752  (achieved/commanded vx, walking holds; n=47)
schedule_completion_rate=0.44
terminations={'schedule_complete': 44, 'fall': 56}

by hold:
     stand: cmd 0.00 -> achieved 0.040 m/s | tripod median=0.37868547429890564 (n=95)
     creep: cmd 0.25 -> achieved 0.184 m/s | tripod median=0.5355641504708648 (n=46)
       low: cmd 0.35 -> achieved 0.168 m/s | tripod median=0.363344786604548 (n=45)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
