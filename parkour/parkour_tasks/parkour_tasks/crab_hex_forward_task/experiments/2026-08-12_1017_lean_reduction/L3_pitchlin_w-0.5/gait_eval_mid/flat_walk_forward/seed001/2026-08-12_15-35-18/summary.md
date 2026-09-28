```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1017_lean_reduction/L3_pitchlin_w-0.5/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_10-25-29/model_21000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40423597190542765  p25=0.39876207810840175  p75=0.4130504232717729
tippy_tap_fraction median=0.07202713487629689
slip_ratio         median=0.0249937617373187
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.49771934814381036 (n=10)
       mid: tripod median=0.4158924676156635 (n=10)
      high: tripod median=0.3003764385327164 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
