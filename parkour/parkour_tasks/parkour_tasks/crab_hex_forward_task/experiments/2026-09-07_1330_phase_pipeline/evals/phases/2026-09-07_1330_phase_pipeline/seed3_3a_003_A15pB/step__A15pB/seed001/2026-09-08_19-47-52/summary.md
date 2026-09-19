```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4762995081342458  p25=0.4210687141248592  p75=0.5360179808890326
tippy_tap_fraction median=0.252222945131208
slip_ratio         median=0.20244060909109315
tracking_ratio     median=0.5803147289096936  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.71
terminations={'schedule_complete': 71, 'fall': 29}

by hold:
     stand: cmd 0.00 -> achieved 0.120 m/s | tripod median=0.47683482352644935 (n=100)
     creep: cmd 0.25 -> achieved 0.163 m/s | tripod median=0.5152468258939349 (n=96)
       low: cmd 0.35 -> achieved 0.159 m/s | tripod median=0.4497332680820493 (n=77)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
