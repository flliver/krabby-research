```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5807709365246483  p25=0.5567511550994231  p75=0.6223621076294827
tippy_tap_fraction median=0.24408862409234777
slip_ratio         median=0.24066823442832558
tracking_ratio     median=0.4521358681581397  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.34
terminations={'schedule_complete': 34, 'fall': 66}

by hold:
     stand: cmd 0.00 -> achieved 0.069 m/s | tripod median=0.7205725268209182 (n=100)
     creep: cmd 0.25 -> achieved 0.121 m/s | tripod median=0.5222994026718172 (n=99)
       low: cmd 0.35 -> achieved 0.134 m/s | tripod median=0.4324425329877182 (n=64)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
