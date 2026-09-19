```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_18-16-56/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5162083365543915  p25=0.4494988979724916  p75=0.5657353529258271
tippy_tap_fraction median=0.21660056361548896
slip_ratio         median=0.21684144021626492
tracking_ratio     median=0.3910194189949648  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.52
terminations={'fall': 48, 'schedule_complete': 52}

by hold:
     stand: cmd 0.00 -> achieved 0.049 m/s | tripod median=0.545411167109429 (n=100)
     creep: cmd 0.25 -> achieved 0.093 m/s | tripod median=0.5239127320171055 (n=99)
       low: cmd 0.35 -> achieved 0.134 m/s | tripod median=0.41339167355326406 (n=81)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
