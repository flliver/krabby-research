```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-28_13-42-11/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.30959049934273286  p25=0.230080957169217  p75=0.3550712461667988
tippy_tap_fraction median=0.34566876866910057
slip_ratio         median=0.3475839222260614
tracking_ratio     median=0.5530620625164417  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.57
terminations={'fall': 43, 'schedule_complete': 57}

by hold:
     stand: cmd 0.00 -> achieved 0.017 m/s | tripod median=0.02279165486663351 (n=100)
     creep: cmd 0.25 -> achieved 0.153 m/s | tripod median=0.4771297330862676 (n=98)
       low: cmd 0.35 -> achieved 0.161 m/s | tripod median=0.4475243135321624 (n=71)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
