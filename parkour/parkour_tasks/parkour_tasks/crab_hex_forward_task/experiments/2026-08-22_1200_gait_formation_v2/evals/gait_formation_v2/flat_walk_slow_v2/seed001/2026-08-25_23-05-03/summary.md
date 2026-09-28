```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-25_16-34-38/model_19996.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5503790083861988  p25=0.5235161288621474  p75=0.5876171409352698
tippy_tap_fraction median=0.2215568862275449
slip_ratio         median=0.2264338738123116
tracking_ratio     median=0.44940279480997003  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.85
terminations={'schedule_complete': 85, 'fall': 15}

by hold:
     stand: cmd 0.00 -> achieved 0.037 m/s | tripod median=0.6252497053425559 (n=96)
     creep: cmd 0.25 -> achieved 0.136 m/s | tripod median=0.5651342549814855 (n=96)
       low: cmd 0.35 -> achieved 0.122 m/s | tripod median=0.5112504183216915 (n=88)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
