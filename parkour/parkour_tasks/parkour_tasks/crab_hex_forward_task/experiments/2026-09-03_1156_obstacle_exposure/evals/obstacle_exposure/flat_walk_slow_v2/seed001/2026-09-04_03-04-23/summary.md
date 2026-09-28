```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_20-42-17/model_24995.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5726812064708902  p25=0.5227794373641105  p75=0.6074549246594536
tippy_tap_fraction median=0.2824836048233552
slip_ratio         median=0.24407080694814814
tracking_ratio     median=0.5212025747763838  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.077 m/s | tripod median=0.6107350842033477 (n=99)
     creep: cmd 0.25 -> achieved 0.144 m/s | tripod median=0.5960746265316442 (n=99)
       low: cmd 0.35 -> achieved 0.163 m/s | tripod median=0.5349210938341249 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
