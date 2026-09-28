```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b5_tripod_v2_w0.15/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_10-34-00/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4080225032491255  p25=0.39727962982443754  p75=0.42288565102147574
tippy_tap_fraction median=0.06109264813163654
slip_ratio         median=0.022074019577886378
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5081631375602991 (n=10)
       mid: tripod median=0.43467407126861257 (n=10)
      high: tripod median=0.2812491273092239 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
