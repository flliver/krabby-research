```
=== crab-hex gait eval ===
scenario   : step__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6045011274148002  p25=0.5509125423902159  p75=0.6394206766960837
tippy_tap_fraction median=0.2548459710700738
slip_ratio         median=0.255739651987943
tracking_ratio     median=0.39431083857535265  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.79
terminations={'schedule_complete': 79, 'fall': 21}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.7174748508922771 (n=100)
     creep: cmd 0.25 -> achieved 0.113 m/s | tripod median=0.5481137755955235 (n=100)
       low: cmd 0.35 -> achieved 0.113 m/s | tripod median=0.5260727571694686 (n=87)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
