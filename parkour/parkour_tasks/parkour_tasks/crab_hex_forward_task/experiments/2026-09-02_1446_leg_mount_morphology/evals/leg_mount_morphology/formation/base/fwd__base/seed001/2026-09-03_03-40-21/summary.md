```
=== crab-hex gait eval ===
scenario   : fwd__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_21-12-20/model_4999.pt
episodes   : 100  unscored: 13
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4647924521936794  p25=0.4249592948757772  p75=0.5365625611903381
tippy_tap_fraction median=0.35050715707650015
slip_ratio         median=0.2553197156243481
tracking_ratio     median=0.466416707842953  (achieved/commanded vx, walking holds; n=87)
schedule_completion_rate=0.01
terminations={'fall': 99, 'schedule_complete': 1}

by hold:
       low: cmd 0.30 -> achieved 0.154 m/s | tripod median=0.5476711972240353 (n=87)
       mid: cmd 0.47 -> achieved 0.147 m/s | tripod median=0.3329086794631845 (n=57)
      high: cmd 0.65 -> achieved 0.178 m/s | tripod median=0.2703922762059256 (n=13)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
