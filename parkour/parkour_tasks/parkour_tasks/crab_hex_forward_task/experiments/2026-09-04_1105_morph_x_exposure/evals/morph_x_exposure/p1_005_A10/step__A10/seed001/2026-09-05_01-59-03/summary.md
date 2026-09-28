```
=== crab-hex gait eval ===
scenario   : step__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_19-33-30/model_4999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3451350006970729  p25=0.3161978487340918  p75=0.4007049143153931
tippy_tap_fraction median=0.30864197530864196
slip_ratio         median=0.21722854681094228
tracking_ratio     median=0.7092611620702395  (achieved/commanded vx, walking holds; n=85)
schedule_completion_rate=0.16
terminations={'fall': 84, 'schedule_complete': 16}

by hold:
     stand: cmd 0.00 -> achieved 0.087 m/s | tripod median=0.37644927398952244 (n=99)
     creep: cmd 0.25 -> achieved 0.183 m/s | tripod median=0.34298973370506064 (n=84)
       low: cmd 0.35 -> achieved 0.128 m/s | tripod median=0.2709786502890803 (n=25)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
