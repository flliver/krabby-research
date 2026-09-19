```
=== crab-hex gait eval ===
scenario   : step__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6230352710848954  p25=0.5677063106175251  p75=0.6577916646124378
tippy_tap_fraction median=0.25855033035367275
slip_ratio         median=0.2569145859193652
tracking_ratio     median=0.35496075955811524  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.7121041068961804 (n=100)
     creep: cmd 0.25 -> achieved 0.101 m/s | tripod median=0.5872668998604622 (n=100)
       low: cmd 0.35 -> achieved 0.106 m/s | tripod median=0.5759699584627296 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
