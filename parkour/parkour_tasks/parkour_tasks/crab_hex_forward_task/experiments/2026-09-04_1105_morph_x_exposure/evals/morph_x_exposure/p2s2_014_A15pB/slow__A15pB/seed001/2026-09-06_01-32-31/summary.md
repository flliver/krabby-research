```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_19-18-29/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.589300421805045  p25=0.5589786040819922  p75=0.6163002294174148
tippy_tap_fraction median=0.2310096153846154
slip_ratio         median=0.2007904098957417
tracking_ratio     median=0.6218814247319724  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}

by hold:
     stand: cmd 0.00 -> achieved 0.083 m/s | tripod median=0.5943479943542648 (n=100)
     creep: cmd 0.25 -> achieved 0.171 m/s | tripod median=0.6251259776457594 (n=99)
       low: cmd 0.35 -> achieved 0.192 m/s | tripod median=0.5755168430734674 (n=95)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
