```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_22-47-02/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4446623891737166  p25=0.4267546953302194  p75=0.46527242762050186
tippy_tap_fraction median=0.2900888265544647
slip_ratio         median=0.3088584734823985
tracking_ratio     median=0.5074488377562003  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.101601549505445 (n=100)
     creep: cmd 0.25 -> achieved 0.142 m/s | tripod median=0.6226626937444919 (n=100)
       low: cmd 0.35 -> achieved 0.153 m/s | tripod median=0.6281208474865201 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
