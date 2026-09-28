```
=== crab-hex gait eval ===
scenario   : fwd__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5487712775231655  p25=0.5288296099745097  p75=0.5689751595189507
tippy_tap_fraction median=0.29618583891255834
slip_ratio         median=0.2895209227921125
tracking_ratio     median=0.2686852910098783  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.81
terminations={'schedule_complete': 81, 'fall': 19}

by hold:
       low: cmd 0.30 -> achieved 0.106 m/s | tripod median=0.6039426068216984 (n=100)
       mid: cmd 0.47 -> achieved 0.123 m/s | tripod median=0.5683021745782979 (n=100)
      high: cmd 0.65 -> achieved 0.127 m/s | tripod median=0.47121328025503895 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
