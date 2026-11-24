# MEAS k1: FL knee interior angle vs knee counts (2026-11-24, Nick at the robot)
Conventions: phone angles in the leg plane, + = outboard end up. Interior angle theta = angle at the knee pin
between the femur (knee->pivot) and the tibia (knee->toe); psi = tibia direction knee->toe, deg CCW from outboard.
theta = psi - (180 + femur).

| # | hip | knee | femur (phone) | tibia readings | toe | psi | theta |
|---|-----|------|---------------|----------------|-----|-----|-------|
| 1 | 482 | 296 | +8.4 (centred) | right-angle square on each tibia edge: -13 and -19 (= edges 13 and 19 deg from vertical; 6 deg taper) | mostly down, slightly inboard | 254 (251-257) | 65.6 (62.6-68.6) |

Cross-checks from the ruler poses with the measured pivot (3.75 out, 2.5 above the belly), femur 23 in:
theta(0) ~ 18.9 (pose A), theta(157) ~ 39.7 (pose B, femur +40.6) and 40.5 (pose C, femur ~+9.0); knee-pin-to-toe 30.9-33.0 in.
| 2 | 0 | 296 | +40.5 (centred) | square on each edge: +17 and +22.5 (= edges 17 and 22.5 deg from vertical; 5.5 deg taper) | down, slightly outboard | 289.75 (287-292.5) | 69.25 (66.5-72.0) |

Hip 0 femur centred = +40.5, so the earlier +40.6 edge reading had no real taper error.
Knee 296 at two hip angles: 65.6 (hip 482) and 69.25 (hip 0), difference 3.65 deg, inside the edge spread. That's consistent with a femur-mounted (uncoupled) knee, and gravity sag would push the difference the other way.
| 3 | 0 | 508 | +40.0 (centred) | phone directly along each edge: 42 and 37.5 below horizontal (4.5 deg taper) | down and outboard | 320.25 (318-322.5) | 100.25 (98.0-102.5) |
| 4 | 0 | 608 | +40.5 (centred) | phone directly along each edge: 30.5 and 26 below horizontal (4.5 deg taper) | outboard, slightly down | 331.75 (329.5-334) | 111.25 (109.0-113.5) |

Hip 0, knee 608: knee pivot to the belly's bottom corner = 27 in diagonal. That can't separate femur 40.5 from 47.6: both give 27.0 in, because the femur points almost straight away from that corner (|P| 4.0 + L 23 = 27 maximum). No contact at hip 0.
| 5 | 0 | 706 | +40.5 (centred) | phone directly along each edge: 19 and 13.5 below horizontal (5.5 deg taper) | outboard, slightly down | 343.75 (341-346.5) | 123.25 (120.5-126.0) |

h3 at hip 0 (knee 706), tape from the floor: hip pivot centre 36.5 in, knee pivot centre 51.5 in, femur 23.0 in pivot to pivot.
That gives a femur angle of asin(15/23) = 40.7 deg (about 39.4-42.0), confirming the phone's 40.5 and ruling out the linkage refit's 47.6.
Hip pivot 2.5 in above the belly (table 34 in tall).
| 6 | 0 | 785 (switch) | +40.5 (centred) | phone directly along each edge: 9.5 and 4 below horizontal (5.5 deg taper) | outboard, slightly down | 353.25 (350.5-356) | 132.75 (130.0-135.5) |

Summary, interior angle vs knee counts:
- derived from ruler poses: 0 -> ~19; 157 -> ~40-44
- measured: 296 -> 65.6 (hip 482) / 69.25 (hip 0); 508 -> 100.25; 608 -> 111.25; 706 -> 123.25; 785 (switch) -> 132.75

h3 at hip 478 (knee 296): phone femur +9.0; knee pivot centre 39.5 in above the floor (hip pivot 36.5), so the femur is asin(3/23) = 7.5 deg by tape.
Hip rotation 0 -> 478: 33.2 deg by tape, 31.5 by phone. The rigid triangle with PF 4.6 / PA 15.9 would give about 46, so the hip linkage numbers are off (the job's free fit says PF about 6.1, PA about 14.6).

Far-end session (knee 170):
- hip 846: phone femur -30.0 hanging, -28.0 with Nick lifting the leg near the knee (2 deg of compliance under the leg's weight).
- Knee pivot 24.5 in above the floor, so the femur is -31.4 by tape.
- Hip rod exposed 7.25 in (1.125 at count 0), so 6.125 in over 846 counts = 5.44 counts/mm. The hip scale matches the knee's 5.33; that's not the cause.
- hip 1069 (extend switch, current 13 -> 7), knee 170:
  - Square on the femur reads 25, knee end down and outboard, so the femur is -65.
  - Knee pivot 15 in above the floor, so -69.2 by tape. At this steepness the tape is about 6 deg/in sensitive, and the floor is a mat over gravel and may not be level, so prefer the square: about -65 to -67.
  - Rod length and the hand-lift test at 1069 not taken.
