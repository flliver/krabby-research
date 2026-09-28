```
=== crab-hex gait eval ===
scenario   : slow__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6011344784773669  p25=0.5732364935825293  p75=0.6288359516962474
tippy_tap_fraction median=0.22777777777777777
slip_ratio         median=0.19448257216357367
tracking_ratio     median=0.438741197227308  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}

by hold:
     stand: cmd 0.00 -> achieved 0.063 m/s | tripod median=0.6651262201307706 (n=100)
     creep: cmd 0.25 -> achieved 0.116 m/s | tripod median=0.5775962765370003 (n=100)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.5692936251417212 (n=95)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
