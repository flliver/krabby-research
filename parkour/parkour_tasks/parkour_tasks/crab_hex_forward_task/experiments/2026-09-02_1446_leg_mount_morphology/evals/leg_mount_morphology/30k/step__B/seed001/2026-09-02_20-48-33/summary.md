```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5965076719383569  p25=0.5413681759173683  p75=0.6312206078553422
tippy_tap_fraction median=0.24924242424242424
slip_ratio         median=0.24544041793613966
tracking_ratio     median=0.39891950118028846  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.7
terminations={'schedule_complete': 70, 'fall': 30}

by hold:
     stand: cmd 0.00 -> achieved 0.068 m/s | tripod median=0.7190340446835678 (n=100)
     creep: cmd 0.25 -> achieved 0.120 m/s | tripod median=0.5230555778098043 (n=99)
       low: cmd 0.35 -> achieved 0.115 m/s | tripod median=0.4994201683328845 (n=77)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
