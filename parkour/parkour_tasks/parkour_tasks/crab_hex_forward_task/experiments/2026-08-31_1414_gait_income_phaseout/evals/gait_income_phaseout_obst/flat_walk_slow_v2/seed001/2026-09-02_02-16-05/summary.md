```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_19-40-14/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6057419259577732  p25=0.5621425132751479  p75=0.6397065965195224
tippy_tap_fraction median=0.23801717920136384
slip_ratio         median=0.20355495842083154
tracking_ratio     median=0.4938913108540045  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.41
terminations={'schedule_complete': 41, 'fall': 59}

by hold:
     stand: cmd 0.00 -> achieved 0.066 m/s | tripod median=0.6906408786505005 (n=100)
     creep: cmd 0.25 -> achieved 0.134 m/s | tripod median=0.5515991946350202 (n=99)
       low: cmd 0.35 -> achieved 0.135 m/s | tripod median=0.4819321768310686 (n=55)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
