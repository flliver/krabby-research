```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3626949805797749  p25=0.32695130279215256  p75=0.3996201091286788
tippy_tap_fraction median=0.38416815742397137
slip_ratio         median=0.35633806806806034
tracking_ratio     median=0.6523550430320949  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.146 m/s | tripod median=0.4064188875717919 (n=100)
     creep: cmd 0.25 -> achieved 0.162 m/s | tripod median=0.24858035722296945 (n=85)
       low: cmd 0.35 -> achieved 0.127 m/s | tripod median=0.17087134937205312 (n=5)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
