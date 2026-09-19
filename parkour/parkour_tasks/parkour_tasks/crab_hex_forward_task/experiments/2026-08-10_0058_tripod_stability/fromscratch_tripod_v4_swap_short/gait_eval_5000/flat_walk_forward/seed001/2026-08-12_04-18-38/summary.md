```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/fromscratch_tripod_v4_swap_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_23-23-51/model_4998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.1809021101388929  p25=0.15570846130688687  p75=0.2093708445782606
tippy_tap_fraction median=0.11134300983461404
slip_ratio         median=0.03847203781447395
schedule_completion_rate=0.7
terminations={'schedule_complete': 7, 'fall': 3}

by hold:
       low: tripod median=0.15489527314419382 (n=10)
       mid: tripod median=0.20994051145210707 (n=9)
      high: tripod median=0.16723987862118597 (n=8)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
