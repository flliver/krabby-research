```
=== crab-hex gait eval ===
scenario   : fwd__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_04-53-36/model_4999.pt
episodes   : 100  unscored: 10
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4720401327849709  p25=0.4135892057951787  p75=0.5097259641361813
tippy_tap_fraction median=0.38348416289592757
slip_ratio         median=0.24439195435805794
tracking_ratio     median=0.528773586162849  (achieved/commanded vx, walking holds; n=91)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
       low: cmd 0.30 -> achieved 0.180 m/s | tripod median=0.5087297488136642 (n=90)
       mid: cmd 0.47 -> achieved 0.200 m/s | tripod median=0.329742698085395 (n=61)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
