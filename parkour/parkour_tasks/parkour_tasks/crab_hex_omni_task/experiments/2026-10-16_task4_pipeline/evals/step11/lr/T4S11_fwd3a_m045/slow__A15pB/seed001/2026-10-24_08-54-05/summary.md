```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.507682041071523  p25=0.45698034096217  p75=0.5366804101797998
tippy_tap_fraction median=0.25302318964290793
slip_ratio         median=0.2092878399585592
tracking_ratio     median=0.6583418728760205  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.76
terminations={'fall': 24, 'schedule_complete': 76}

by hold:
     stand: cmd 0.00 -> achieved 0.142 m/s | tripod median=0.548926546649986 (n=100)
     creep: cmd 0.25 -> achieved 0.194 m/s | tripod median=0.5177967164400313 (n=98)
       low: cmd 0.35 -> achieved 0.182 m/s | tripod median=0.45453399323541754 (n=84)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
