```
=== crab-hex gait eval ===
scenario   : fwd__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5241047823571585  p25=0.5064997244326023  p75=0.5468398291171239
tippy_tap_fraction median=0.29960903215511053
slip_ratio         median=0.2769688572507627
tracking_ratio     median=0.2779268251308162  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.69
terminations={'fall': 31, 'schedule_complete': 69}

by hold:
       low: cmd 0.30 -> achieved 0.110 m/s | tripod median=0.5879135160176002 (n=100)
       mid: cmd 0.47 -> achieved 0.127 m/s | tripod median=0.5324717167121994 (n=99)
      high: cmd 0.65 -> achieved 0.135 m/s | tripod median=0.4514196851225303 (n=96)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
