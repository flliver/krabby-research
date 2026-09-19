```
=== crab-hex gait eval ===
scenario   : fwd__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_07-16-23/model_4999.pt
episodes   : 100  unscored: 8
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4657338198462697  p25=0.40056344855438386  p75=0.5051927200774528
tippy_tap_fraction median=0.3521462639109698
slip_ratio         median=0.23901702849957857
tracking_ratio     median=0.578878112110223  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
       low: cmd 0.30 -> achieved 0.209 m/s | tripod median=0.5013088887560526 (n=92)
       mid: cmd 0.47 -> achieved 0.214 m/s | tripod median=0.24872953479681226 (n=71)
      high: cmd 0.65 -> achieved 0.100 m/s | tripod median=0.026876227485737147 (n=1)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
