```
=== crab-hex gait eval ===
scenario   : step__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_07-16-23/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.27048643888160906  p25=0.22849723445195094  p75=0.30317678085231453
tippy_tap_fraction median=0.4356959424756035
slip_ratio         median=0.43820216787728206
tracking_ratio     median=0.7297224339688728  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.34
terminations={'fall': 66, 'schedule_complete': 34}

by hold:
     stand: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.05076570081962821 (n=100)
     creep: cmd 0.25 -> achieved 0.225 m/s | tripod median=0.494351789012846 (n=99)
       low: cmd 0.35 -> achieved 0.164 m/s | tripod median=0.3101287494316198 (n=58)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
