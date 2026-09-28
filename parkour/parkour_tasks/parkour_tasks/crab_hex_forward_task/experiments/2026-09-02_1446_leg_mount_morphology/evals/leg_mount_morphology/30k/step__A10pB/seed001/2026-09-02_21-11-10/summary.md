```
=== crab-hex gait eval ===
scenario   : step__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5989771456129528  p25=0.5430539301655872  p75=0.6387322048172672
tippy_tap_fraction median=0.2573442010528103
slip_ratio         median=0.2626143236790157
tracking_ratio     median=0.3609740046580728  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.066 m/s | tripod median=0.7279034795860846 (n=100)
     creep: cmd 0.25 -> achieved 0.104 m/s | tripod median=0.5651770006573932 (n=100)
       low: cmd 0.35 -> achieved 0.107 m/s | tripod median=0.5380475885333516 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
