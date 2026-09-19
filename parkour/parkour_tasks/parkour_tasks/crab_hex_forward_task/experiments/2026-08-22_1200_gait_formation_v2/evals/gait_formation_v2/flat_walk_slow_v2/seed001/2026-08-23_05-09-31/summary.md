```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_23-15-14/model_4999.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.02649840582001283  p25=0.014991713264817099  p75=0.0496033292142159
tippy_tap_fraction median=0.38335927925540725
slip_ratio         median=0.5931438464206993
tracking_ratio     median=0.07717174176350416  (achieved/commanded vx, walking holds; n=82)
schedule_completion_rate=0.25
terminations={'fall': 75, 'schedule_complete': 25}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.025271510897320146 (n=96)
     creep: cmd 0.25 -> achieved 0.013 m/s | tripod median=0.02466143395608504 (n=82)
       low: cmd 0.35 -> achieved 0.020 m/s | tripod median=0.014699970823795335 (n=51)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
