```
=== crab-hex gait eval ===
scenario   : slow__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_21-12-20/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5370083390168021  p25=0.49758080306265096  p75=0.5751192377011991
tippy_tap_fraction median=0.27652267461105595
slip_ratio         median=0.28628092804164246
tracking_ratio     median=0.4264894154009813  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.83
terminations={'schedule_complete': 83, 'fall': 17}

by hold:
     stand: cmd 0.00 -> achieved 0.039 m/s | tripod median=0.4581297228402467 (n=100)
     creep: cmd 0.25 -> achieved 0.114 m/s | tripod median=0.6356570811923139 (n=100)
       low: cmd 0.35 -> achieved 0.136 m/s | tripod median=0.5167598136383488 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
