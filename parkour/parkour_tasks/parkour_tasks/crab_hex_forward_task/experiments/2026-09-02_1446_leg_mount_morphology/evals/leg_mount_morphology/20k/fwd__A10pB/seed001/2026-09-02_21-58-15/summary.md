```
=== crab-hex gait eval ===
scenario   : fwd__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5681391649005467  p25=0.5488523431727187  p75=0.591387128447838
tippy_tap_fraction median=0.27501563477173235
slip_ratio         median=0.2538351778547414
tracking_ratio     median=0.3117362086156604  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
       low: cmd 0.30 -> achieved 0.125 m/s | tripod median=0.6477727976771437 (n=100)
       mid: cmd 0.47 -> achieved 0.136 m/s | tripod median=0.5800317125471579 (n=100)
      high: cmd 0.65 -> achieved 0.147 m/s | tripod median=0.49197999936500797 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
