```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.37948632134087357  p25=0.35841494572915444  p75=0.40800698555006765
tippy_tap_fraction median=0.35743321718931476
slip_ratio         median=0.3193370475454701
tracking_ratio     median=0.6104520858108916  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.08
terminations={'fall': 92, 'schedule_complete': 8}

by hold:
     stand: cmd 0.00 -> achieved 0.156 m/s | tripod median=0.4723680021266776 (n=100)
     creep: cmd 0.25 -> achieved 0.163 m/s | tripod median=0.28946118192354914 (n=100)
       low: cmd 0.35 -> achieved 0.140 m/s | tripod median=0.2536439611245173 (n=37)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
