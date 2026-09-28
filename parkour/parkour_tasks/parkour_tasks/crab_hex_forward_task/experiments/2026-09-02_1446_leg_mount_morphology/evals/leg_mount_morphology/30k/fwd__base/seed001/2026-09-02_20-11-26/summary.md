```
=== crab-hex gait eval ===
scenario   : fwd__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 10
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4302673164701526  p25=0.3859264604794488  p75=0.4711704777861543
tippy_tap_fraction median=0.3413199426111908
slip_ratio         median=0.23569126584240643
tracking_ratio     median=0.45685331742307994  (achieved/commanded vx, walking holds; n=91)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
       low: cmd 0.30 -> achieved 0.147 m/s | tripod median=0.4656240787368193 (n=90)
       mid: cmd 0.47 -> achieved 0.163 m/s | tripod median=0.31292588415748046 (n=46)
      high: cmd 0.65 -> achieved 0.179 m/s | tripod median=0.29417903395459305 (n=6)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
