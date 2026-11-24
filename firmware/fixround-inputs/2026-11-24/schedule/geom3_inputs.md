# FL leg geometry measured 2026-11-24, for the table-safe schedule rerun (geom3)

Frame: leg plane. x = horizontal distance out from the body's side wall (+ outboard), z = height above the belly plane (+ up).
Belly plane = table top. Floor is 34 in below it. Units are inches and degrees.

## Stand
- Table: 20.5 in wide (across the leg rows), 34 in tall, longer than the robot front to back. Partly hollow (supports), so treat it as solid.
- The robot sits centred on its belly. The body is 28 x 48 in, so it overhangs the table by about 13.75 in each side.
- So the table's side edge is about 13.75 in INBOARD of the side wall (x = -13.75), from z = 0 down to the floor (z = -34).

## Hip pivot and plate
- Hip (femur) pivot P: x = +3.0, z = +2.5 (tape: 36.5 in above the floor). The diagonal from the body's bottom corner gives z = 2.65.
- Hip actuator anchor A: about 1.5 in inboard of P, about 15.8 in above P. The A-P line is 5.5 deg off vertical (top toward the body), in the leg plane.
- Hip plate outline (MEAS p2):
  - its bottom is flush with the belly (0 to 0.25 in above it), so nothing hangs below the belly;
  - its top is 24 in above the belly plane;
  - its outer edge is about 5.5 in outboard of the side wall, about 2.5 in past the pivot.
  - The CAD had it 6.6 in below the belly, which made the sim's curled tibia hit the plate first.

## Femur
- 23.0 in pivot to pivot (measured).
- Hip rod eye pin: 4.75 in from P (inner hole). The outer hole (5.86 in from P, 17.14 from the knee) carries the knee actuator's anchor.
- Femur elevation (+ = knee end up) against hip counts:
  - 0: +40.7 (tape) / +40.5 (phone, centred);
  - 478: +7.5 (tape) / +9.0 (phone);
  - 482: +8.4 (phone);
  - 612: -3.5 (phone);
  - 846: -30.0 (phone, hanging; -28.0 with the leg's weight lifted by hand, i.e. 2 deg of compliance) / -31.4 (tape);
  - 1069 (extend switch): -65 (right-angle square, +/-2) / -69.2 (tape; the garage floor is a mat over gravel, and the tape is 6 deg/in sensitive here, so prefer -65).
  - Pose D (hip 1069, knee 157, toe at x -16.5, z -6) agrees: with femur -65 and knee about 40 deg, the toe comes out at about (-16.7, -4.6).
- Hip rod exposed length: 1.125 in at count 0, 7.25 in at 846, 9.0 in at 1069. The actuator label says a 203.2 mm stroke, so the hip is 5.26 counts/mm (ruler: 5.34-5.44); the knee is 5.33.
- Open: the pinned triangle (PA 15.9, PF 4.75) predicts at least 41-44 deg over 0 -> 478, but the femur turns 32-33. The effective lever is about 6.1 in. Use the MEASURED angles, not the triangle.

## Knee
- Knee interior angle (knee->hip pivot vs knee->toe) against knee counts, +/-2 deg:
  - 0: about 19 (pose A)
  - 157: about 40-44 (poses B/C; the job's refit says 48.5)
  - 296: 65.6-69.25
  - 508: 100.25
  - 608: 111.25
  - 706: 123.25
  - 785 (switch): 132.75
- The retrain job's refit: 19 / 48.5 / 68 / 93 / 105-106 / 132-133 at 0 / 157 / 300 / 500 / 600 / 779-785.
- The knee is femur-mounted: its angle doesn't depend on the hip (66-69 at 296 at hip 482 vs hip 0).
- Knee pin to toe tip: about 32.5 in (repo; ruler poses give 31-33).
- The tibia is tapered symmetrically; edges are 4.5-6 deg apart.

## Ruler poses (cam yaw unknown but fixed; side readings may need the leg-plane yaw, refit about 22 deg)

| Pose | Hip | Knee | x (in) | z (in) |
|---|---|---|---|---|
| A | 0 | 0 | +5 | -10 |
| B | 0 | 157 | +16 | -13 |
| C | 475 | 157 | +5 | -19 |
| D | 1069 | 157 | -16.5 | -6 |

## Contact facts
- With the knee at its curl end (0-24), hip extension presses the tibia into the belly at hip 941-1003.
- With the knee at 157-175, the full hip sweep 0-1069 cleared the belly on FL and FR. At 1069 the toe was 16.5 in under the body, 6 in below the belly: inside the table footprint, so it only cleared because this table is hollow there.
- No contact at hip 0 with any knee setting.

## Not yet measured
- Plan-view yaw (MEAS y1).
- The hip plate outline (MEAS p2).
- Where the femur's hinge axis really is. The femur behaves as if pinned with a lever of about 6.1 in, not the measured 4.75 in from the bolt. Use the MEASURED femur angles, not the triangle.

## Schedule end-pose check on the robot (2026-11-24 17:30, FL, knee 264, hip 866, this garage table)
- Toe about 25 in below the belly (26 by the 8 in floor reading), about 8 in above the floor, 27 in diagonal from the body's bottom corner.
- **Toe about 5 in from the table's side face, measured horizontally ("a bit hard to measure").**
- Model (v3, nominal) predicted the toe at about (-5, -25.2): 8-9 in from a centred table edge, 9 in above the floor.
- So the real toe is about 3-4 in further inboard than the nominal model at this pose. That is beyond the ensemble worst case (-8.3) if the robot is centred, or explained if the robot sits about 2 in off-centre toward FL. Height agrees.
- Clear of a solid table by about 5 in, but the table margin is thinner than modelled. Feed this into the next schedule pass, e.g. H about 820-840 for more table margin.
