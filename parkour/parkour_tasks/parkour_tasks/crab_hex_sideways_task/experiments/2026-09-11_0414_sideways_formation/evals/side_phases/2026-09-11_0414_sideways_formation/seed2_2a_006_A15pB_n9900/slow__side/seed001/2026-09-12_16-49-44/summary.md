```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9900.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4618061441362915  p25=0.4352238828284934  p75=0.4912432520631031
tippy_tap_fraction median=0.38153310104529614
slip_ratio         median=0.4077438338281031
tracking_ratio     median=0.7593869061326282  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=92)
     creep: cmd 0.25 -> achieved 0.223 m/s | tripod median=0.6710043713283994 (n=99)
       low: cmd 0.35 -> achieved 0.225 m/s | tripod median=0.7012728233745891 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
