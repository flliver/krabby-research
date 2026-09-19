```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-09_02-06-51/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4586913016108718  p25=0.4064422508991343  p75=0.5313330156942091
tippy_tap_fraction median=0.24918300653594772
slip_ratio         median=0.20161072070514952
tracking_ratio     median=0.5459416328297306  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.68
terminations={'fall': 32, 'schedule_complete': 68}

by hold:
     stand: cmd 0.00 -> achieved 0.124 m/s | tripod median=0.4801085633649707 (n=100)
     creep: cmd 0.25 -> achieved 0.155 m/s | tripod median=0.45726069050885443 (n=92)
       low: cmd 0.35 -> achieved 0.157 m/s | tripod median=0.4265251871090688 (n=73)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
