```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6447213179971858  p25=0.6066985695491975  p75=0.6620345319274876
tippy_tap_fraction median=0.23053613053613053
slip_ratio         median=0.2007268361218011
tracking_ratio     median=0.4184815805437416  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.062 m/s | tripod median=0.6794777316575609 (n=100)
     creep: cmd 0.25 -> achieved 0.109 m/s | tripod median=0.6247027583381746 (n=100)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.6339109231153037 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
