```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-04_23-34-31/model_19999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.07439551582998175  p25=0.06535553853005971  p75=0.0924378553216962
tippy_tap_fraction median=0.35929129088706424
slip_ratio         median=0.034091634830901385
schedule_completion_rate=0.9
terminations={'schedule_complete': 9, 'fall': 1}

by hold:
       low: tripod median=0.06098817637748846 (n=10)
       mid: tripod median=0.07819630351008508 (n=9)
      high: tripod median=0.08435620896111302 (n=9)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
