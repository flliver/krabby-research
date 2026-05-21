# krab Motor Sourcing Summary

This document tracks motor and linear actuator sourcing for the krab hexapod platform across the two primary joint categories: **hip yaw** (continuous-rotation gearmotors) and **hip pitch / knee** (linear actuators).

All suppliers are based in mainland China and contacted via Alibaba. Quotes reflect sample/prototype-quantity pricing; production pricing TBD pending sample evaluation.

---

## Application Context

- **Platform:** 6-leg ripple gait hexapod, ~150 lb body weight, ~800-950 lb target operating weight
- **Joint count per robot:** 18 actuators total
  - 6 × hip yaw (rotational)
  - 6 × hip pitch (linear)
  - 6 × knee (linear)
- **Power architecture:** 24V / 100Ah pack, ~2,040 Wh usable, ~600W average walking budget
- **Control:** Custom Arduino Mega shield, dual H-bridge boards per leg, hall encoder/potentiometer feedback on PCINT pins, JST XA (v0.1) or JST VHR-6N (v0.2) connectors from motors to power boards.

---

## Sourcing Philosophy

Every joint type goes through the same evaluation pattern: **specify a primary supplier for the production build, then bakeoff against 1–2 alternates** at sample quantity to validate the production decision before committing to 100+ units in late 2026.

The alternates are not interchangeable products — they're chosen to *falsify specific hypotheses* about the primary pick (e.g., "is brushless durability worth the price premium?", "does ball-screw efficiency translate to better load-speed performance?", "does higher load margin reduce wear under our duty cycle?").

The goal is to build 3-5 robots in May/June 2026, then put those robots in the field, see which motor combination works the best and has the best price/value, then double down on that motor type and iterate as needed.

---

## Hip Yaw Motors (Rotational)

The hip yaw joint rotates the leg around the vertical axis at the body attachment point. Sourced as gearmotors (motor + integrated planetary gearbox + encoder/hall feedback).

**Target spec at the joint:**
- Output torque: ≥ 16 N·m (drives leg + foot mass through swing arc, plus margin for body inertia during turns)
- Output speed: ~30 RPM nominal (matches gait cycle time at expected body translation speeds)
- 24V DC, integrated encoder, ≤120W input power class
 
### Primary supplier: Zhejiang Xingli (Allen Hu) ⭐ ORDERED

90mm × 90mm motor frame, 120W class, 24V brushed DC with 5GU-series planetary gearbox. Output specs match the target joint envelope cleanly.

**Order summary:**

| Qty | Variant | Motor | Output speed | Output torque | Purpose |
|---|---|---|---|---|---|
| 6 | Brushed primary | 90mm 120W brushed, 1800 RPM rated | 30 RPM | 20 N·m | **Production hip yaw build (1 robot)** |
| 1 | Brushed fast variant | Same motor frame, lower reduction | 90 RPM | ~7 N·m | Speed-tradeoff bakeoff |
| 1 | Brushless variant | 90mm 120W BLDC, 3000 RPM rated | 30 RPM | 20 N·m | Brushless durability bakeoff |

**Datasheet:** [`Zhejiang_Xingli_-_Z5D120-24GU-30S-5GU100KB-Model.pdf`](./Zhejiang_Xingli_-_Z5D120-24GU-30S-5GU100KB-Model.pdf) (representative — actual ordered units have variant-specific gearbox ratios)

**Why this won:**

1. **Right size at the right power class.** 90mm/120W is the standard industrial frame for this torque-speed envelope. Larger (130mm/200W+) is overkill and adds significant weight at the body attachment point; smaller (60mm/40W) doesn't deliver the torque needed to drive the leg through the swing arc against body-inertia loading during turns.
2. **Standard 5GU planetary gearbox family** has reductions from 1:3 to 1:200, which gave us flexibility to spec the production unit (60:1, 30 RPM out) and the fast variant (lower ratio, 90 RPM out) using the *same motor frame*. Identical mounting flange, shaft, and electrical interface across both variants — drop-in interchangeable for bakeoff testing.
3. **Brush replaceability** is unusual for Chinese industrial motors at this price point. Xingli's Z5 series is documented for external brush access, which means 2,000-hour brush life is *maintainable* rather than terminal — significant for field operation where replacing brushes is far cheaper than swapping the whole gearmotor.
4. **2500-line Ruiying optical encoder** on the brushed unit gives high-resolution position feedback (vs. typical 13–100 PPR hall encoders on competing planetary gearmotors). Resolution at output: 2500 × 60 = 150,000 counts/rev = ~0.0024°/count. Excellent for closed-loop position control.

**Why three variants ordered instead of just six production units:**

The 6 production units commit us to a brushed architecture at 30 RPM output. Before we lock that in for a 36-unit production run in 2026, we need data on:

- **Whether 30 RPM is fast enough** for body translation during gait. If swing-phase repositioning is the rate-limiter, the 90 RPM variant tells us the speed ceiling is worth pursuing — but probably at the cost of torque margin.
- **Whether brushed is good enough** for our duty cycle. The brushless variant is the same motor frame, same mounting, same control interface — but with hall commutation in place of brushes. Run it side-by-side with the 6 brushed units on the bench for 200+ hours and the wear comparison is decisive for the 2026 decision.

This is the same bakeoff structure used for the linear actuators below: commit to a production primary, characterize the alternatives in parallel, decide on production architecture from real data.

**Application caveat:** IP20 protection class on the brushed unit means additional sealing required for outdoor construction-site operation. Plan to pot the encoder, add a labyrinth seal at the shaft, and enclose the brush access plate. Brushless variant is similar.

### Alternate supplier: Dongguan Faradyi (Teana Xie) ⭐ ORDERED

**Proforma Invoice:** [`PI-FDTX20260421_Faradyi.pdf`](./PI-FDTX20260421_Faradyi.pdf) — Total $365.90 incl. DHL freight to USA, 30-day lead time

| Qty | Variant | Model | Output speed | Output torque | Price |
|---|---|---|---|---|---|
| 1 | 30 RPM Brushed Planetary | X60R63126S-130K-24V-30R-12B (1:130) | 30 RPM rated, 24 RPM under load | 20 N·m | $51.20 |
| 1 | 60 RPM Brushed Planetary | X60R68118S-71K-24V-60R-12B (1:71) | 60 RPM rated, 48 RPM under load | 16 N·m | $68.40 |
| 1 | 30 RPM Brushless Planetary | (BLDC + external driver board) | 30 RPM | TBD | $62.90 |
| 3 | L-type Mount Brackets | — | — | — | $5.20 ea |
| 16 | M6×12 Pan Head Phillips screws | — | — | — | $0.05 ea |

**Drawings:** [`faradyi-X60R63126S-130K.png`](./faradyi-X60R63126S-130K.png) (1:130), [`faradyi-X60R68118S-71K.png`](./faradyi-X60R68118S-71K.png) (1:71)

**Why this is the alternate:**

The Faradyi units are a smaller frame (63–68mm vs. Xingli's 90mm) and integrate the planetary gearhead directly into the motor housing rather than as a bolt-on. Three relevant differences from the Xingli production unit:

1. **More compact form factor.** 63–68mm gearbox diameter vs. Xingli's 90mm, ~180mm total length vs. Xingli's ~250mm. If packaging at the body attachment turns out to be tight, Faradyi gives us a smaller-envelope option without losing torque (still 16–20 N·m).
2. **Built-in encoder at 13 PPR.** Lower resolution than Xingli's 2500-line encoder, but tightly integrated and 5V-compatible — simpler wiring, less to break in field service.
3. **Brushed/brushless parity at near-identical price** ($51 brushed vs. $63 brushless, ~$12 BLDC premium). If the Xingli brushless variant comes back substantially more expensive than the brushed ($300+ vs $150), Faradyi offers a low-cost path to brushless at this joint.

The 60 RPM brushed unit serves the same role as Xingli's 90 RPM variant (speed-tradeoff bakeoff at lower torque) but at a different motor size — which lets us evaluate whether the high-speed-low-torque tradeoff lives at the gearbox or at the motor frame.

The 30 RPM brushless unit gives us a second brushless data point alongside Xingli's brushless variant. Same joint configuration, two suppliers, same bakeoff bench rig — apples-to-apples comparison of build quality, encoder behavior, thermal performance, and wear rates.

### Reference / not ordered: Ningbo Leison (Tony Chou)

**Catalog:** [`Ningbo_-_DC_GEAR_MOTOR_CATALOGUE_10.pdf`](./Ningbo_-_DC_GEAR_MOTOR_CATALOGUE_10.pdf) — 5D120 series

Direct competitor to Xingli at the same form factor (90mm, 120W) and broadly comparable specifications. Catalog kept on file as a fallback if Xingli's lead time or quality issues emerge during production. Tony was responsive on the initial inquiry and the Ningbo catalog has the broadest reduction selection of any supplier evaluated (24 reduction ratios from 1:3 to 1:200), but no advantage strong enough to displace Xingli for the bakeoff.

---

## Linear Actuators (Hip Pitch & Knee)

200mm (8") stroke, 24V brushed lead-screw or ball-screw actuators with hall encoder feedback. Sourced for the knee and hip pitch joints, which lift and lower the body during stance phase.

### Sourcing Decision Framework

Application loading is asymmetric across joint types:
- **Knee:** Light load (lever-arm reduces axial force to <500N), benefits from speed for swing-phase clearance
- **Hip pitch:** Heavy load (peak ~1,500N during stance phase), low speed acceptable

This invites a mixed-actuator architecture: lighter/faster units at knee, heavier/slower units at hip pitch.

**Lifetime claim quality matters as much as the number.** Across 6+ suppliers solicited, only Wuxi Yuhuang's engineering team validated their lifetime number against our specific duty load — everyone else quoted catalog defaults. Engineering-validated 50k cycles is a different category of claim than catalog 20k cycles, even though they look comparable on a spec sheet.

### Primary knee supplier: Wuxi Yuhuang YH8-523D ⭐ ORDERED

**Contact:** Simon Zhang (Imdadul, sales)  
**Type:** Brushed DC, trapezoidal screw  
**Datasheet:** [`YH8-523D_Linear_Actuator_Specification.pdf`](./YH8-523D_Linear_Actuator_Specification.pdf)  
**Order:** 12 units @ $53/unit (knee primary build, includes destructive teardown sample)  
**Hall feedback:** +$4/unit

| Spec | Value |
|---|---|
| Voltage | 24V DC |
| Stroke | 200mm |
| Motor diameter | 51mm |
| Load | 500N (B-code: 4500N self-lock, 40:1 reduction, 9mm pitch) |
| No-load speed | 38 mm/s |
| Load speed | 33 mm/s |
| Full-load current | 4.5A |
| Lifetime | **50,000 cycles** (engineering-validated for application load) |
| IP rating | IP65 |
| Operating temp | -40 °C to +65 °C |
| Encoder | Hall, 4-pole, configurable pulse/mm by gearing |

**Why this won:**

1. **Engineering-validated lifetime.** Simon was the only supplier whose engineering team engaged the duty-load question — asking about our application before answering — and returned 50k cycles validated against our specific load profile. Every other supplier quoted catalog defaults of 10–20k cycles, which are not application-conditioned.
2. **Best-in-class durability:price ratio.** 50k cycles at $53 is 5× better lifetime than the next-cheapest comparable quote (Junshi at $55, 10k cycles) for $2 less per unit. Yuhuang's number isn't just bigger, it's defensible.
3. **Adequate speed and load for the knee joint.** 33 mm/s under 500N load and 38 mm/s no-load comfortably handles knee operation: stance-phase joint motion (loaded, slow) and swing-phase toe clearance (unloaded, ~25mm lift in <1 second).

### Primary hip-pitch supplier: Changzhou Sunline SLA08 (2000N variant) ⭐ ORDERING

**Contact:** Ariana Wang  
**Type:** Brushed DC, trapezoidal screw  
**Datasheet:** [`The_detailed_specification_of_SLA08_linear_actuator.pdf`](./The_detailed_specification_of_SLA08_linear_actuator.pdf)  
**Order:** 6 units @ $84/unit (hip-pitch primary build)

| Spec | Value |
|---|---|
| Voltage | 24V DC |
| Stroke | 200mm |
| Motor diameter | 64mm |
| Load | 2000N |
| Self-lock | ~2,800N |
| No-load speed | 35 mm/s |
| Load speed | 28 mm/s |
| Full-load current | 8A |
| No-load current | 1.8A |
| IP rating | IP65 |
| Encoder | Hall |

**Why this won:**

1. **4× load margin at the most-loaded joint.** Hip-pitch sees peak forces around 1,500N during stance phase on an 800 lb robot. The 2000N variant operates *at or below* nameplate during typical loading and ~1.5× nameplate at peak.
2. **Same 64mm motor frame as the Sunline 500N variant.** This is structurally important for the architecture: cross-compatible spares, identical electrical interface, identical mounting envelope. Two different gear ratios, one motor platform.
3. **Speed envelope is fine for hip-pitch.** 28 mm/s under load is a little slow, but hip-pitch is a body-height-adjustment joint operating in stance phase — speed isn't the constraint there. The bottleneck is at the knee for swing-phase clearance, which is handled by separate (Yuhuang/Sunline 500N) actuators.
4. **Same price as the 500N variant ($84).** Sunline pricing is anomalous in a good way, it's the only 2000N motor that reaches close to 30mm/s at this price point, worth the extra $30 for 1500N. 

### Knee bakeoff alternate: Changzhou Sunline SLA08 (500N variant) ⭐ ORDERING

**Contact:** Ariana Wang  
**Type:** Brushed DC, trapezoidal screw configurable  
**Datasheet:** [`The_detailed_specification_of_SLA08_linear_actuator.pdf`](./The_detailed_specification_of_SLA08_linear_actuator.pdf)  
**Order:** 6 units @ $84/unit (knee bakeoff, paired with Yuhuang YH8-523D)

| Spec | Value |
|---|---|
| Voltage | 24V DC |
| Stroke | 200mm (50–1080mm available) |
| Motor diameter | 64mm |
| Load | 500N (low-reduction gearing) |
| No-load speed | 80 mm/s |
| Load speed | TBD (pending load-curve data) |
| IP rating | IP65 |
| Duty cycle | 25% standard |
| Encoder | Hall (configurable pulse/mm) or potentiometer |

**Why this is the knee bakeoff:**

1. **Highest no-load speed in the field (80 mm/s).** 2× the Yuhuang YH8-523D's 38 mm/s no-load. If the load-speed curve preserves any meaningful fraction of that advantage at 500N (TBD pending curve data from Ariana), this gives us substantially faster swing-phase repositioning at the knee so we can experiment with leg speed vs power and see what we like.
2. **Single-supplier architecture if Sunline wins.** Sunline-500N at knees + Sunline-2000N at hip-pitch = one supplier, one motor frame across all 12 linear actuators. Single SKU for spares, identical electrical interface, simpler validation. Worth real money in operational simplicity.
3. **Tests the upper performance bound of brushed lead-screw actuators.** If Sunline 500N delivers as datasheet suggests, it sets the speed ceiling for what's achievable in this product class. If it underperforms its no-load number badly under load, that tells us trapezoidal-screw actuators have a hard physical ceiling around the Yuhuang 33 mm/s and we should stop chasing speed.

### Ball-screw evaluation sample: Wuxi Yuhuang YH8-520P ⭐ ORDERED

**Contact:** Simon Zhang  
**Type:** Brushed DC, **ball screw** (vs. trapezoidal on the YH8-523D)  
**Datasheet:** [`YH8-520P_Linear_Actuator_Specification.pdf`](./YH8-520P_Linear_Actuator_Specification.pdf)  
**Order:** 1 unit @ $160 (architecture evaluation)

| Spec | Value |
|---|---|
| Voltage | 24V DC |
| Stroke | 30–1000mm (customizable) |
| Speed range | 7–50 mm/s (multiple gear options) |
| Max load | 16,000N |
| IP rating | IP66 |
| Operating temp | -40 °C to +65 °C |
| Screw type | Ball screw |

**Why this single sample:**

Ball screws offer ~85–90% mechanical efficiency vs. ~25–35% for trapezoidal screws — meaning *much* better load-speed performance from the same motor power. The downside is they aren't self-locking (back-driveable under load), which matters for static holding without continuous motor current.

A single $160 unit lets us measure:
1. **Real efficiency under our duty cycle.** If ball-screw advantage delivers 50+ mm/s under 500N load (vs. trapezoidal's 33), it's a major architectural unlock for swing-phase speed.
2. **Back-drive behavior.** Whether the joint will sag under static load when motor power is removed, and how much that matters for our gait pattern given the knee is just preventing the hip from sagging under load and shares with the hip, which carries most of the load.
3. **Acoustic and tactile feel.** Ball screws are usually quieter and smoother than trapezoidal — relevant for field operation alongside humans.

If this unit performs strongly, it's the production candidate for 2026. If it's only marginally better, the trapezoidal architecture wins on price ($53 vs. $160).

### Mega Motor Sample: Wuxi Aigwell 300W  ⭐ ORDERED

**Contact:** Jade Zhou — $120/unit, 800N, 65 mm/s no-load, 48 mm/s loaded, 12A, 20k cycles

**Why on hold:** Best raw performance (load × speed) in the field, but two structural concerns:

- **12A continuous current** is 4× the per-actuator continuous budget. 18 actuators at this draw push average walking power past 1kW and require leg-trunk wiring upgrade from XT60H (30A) to XT90 (45A).
- **20k cycles catalog lifetime.** No engineering validation, and at the higher loading this unit enables, wear rates may run worse than the catalog number suggests.
- **It's huge!** Giant motor, giant price, but amazing speed/performance, so we'll see if we really need it. Maybe for a Krabby-Duo.

Single bakeoff unit was considered but deprioritized; the bakeoff budget is better spent on the ball-screw evaluation (Yuhuang 520P) which addresses the same "what's the upper performance bound?" question with broader architectural implications. Aigwell 300W remains on file as an emergency option if extreme torque is needed at a specific joint.


### Held / not ordered: Changzhou Holry Electric

**Contact:** Jerry Xu — $59/unit, 400N, 57 mm/s no-load, 40 mm/s loaded, 20k cycles (catalog)

**Why on hold:** Best speed-under-load in the brushed budget tier, but the lifetime number is generic catalog (not engineering-validated like Yuhuang's), and 400N is below knee/hip-pitch spec. With Yuhuang ordered as primary at the knee and Sunline 500N as the speed alternate, Holry's unique value (high load-speed at low cost) is already covered by the Sunline bakeoff, and it's not worth it to lose the Nm for just $20/motor cheaper. Keeping warm for future production consideration if Yuhuang and Sunline both disappoint.

### Rejected: Dongguan Junshi (Peter Lo)

$55/unit, 500N, 33 mm/s loaded, hall, 10k cycles. **Strictly dominated by Yuhuang YH8-523D** at $53 with same speed/load and 5× the cycle rating. Peter was responsive on customization but the underlying product can't compete.

### Rejected: Wuxi Aigwell base unit (Jade Zhou)

$38/unit, 600N, 25–30 mm/s loaded, 20k cycles. Slow under load and weakest lifetime claim in the field. Cheap but not competitive on the dimensions that matter.

---

## Comparison Matrix — Linear Actuators

| Supplier / Model | Price | Load | No-load Speed | Load Speed | Full-load Current | Lifetime | Lifetime Quality | Status |
|---|---|---|---|---|---|---|---|---|
| Yuhuang YH8-523D | $53 | 500N | 38 mm/s | 33 mm/s | 4.5A | 50k cycles | Engineering-validated | **Ordered (12)** |
| Yuhuang YH8-520P | $160 | up to 16kN | varies | 7–50 mm/s | TBD | TBD | TBD | **Ordered (1)** |
| Sunline SLA08 (500N) | $84 | 500N | 80 mm/s | TBD | TBD | TBD | TBD | **Ordering (6)** |
| Sunline SLA08 (2000N) | $84 | 2000N | 35 mm/s | 28 mm/s | 8A | TBD | TBD | **Ordering (6)** |
| Holry brushed | $59 | 400N | 57 mm/s | 40 mm/s | TBD | 20k cycles | Catalog | Held |
| Aigwell 300W | $120 | 800N | 65 mm/s | 48 mm/s | 12A | 20k cycles | Catalog | **Ordered (1)** |
| Junshi (rejected) | $55 | 500N | 38 mm/s | 33 mm/s | 4.5A | 10k cycles | Catalog | — |
| Aigwell base (rejected) | $38 | 600N | 35.7 mm/s | 25.4–30 mm/s | 3.1A | 20k cycles | Catalog | — |

---

## Bakeoff Plan

| Order | Qty | Unit Price | Total | Purpose |
|---|---|---|---|---|
| **Hip Yaw — Xingli** | | | | |
| Brushed primary (30 RPM, 20 N·m) | 6 | TBD | TBD | Production hip yaw build (1 robot) |
| Brushed fast (90 RPM) | 1 | TBD | TBD | Speed tradeoff bakeoff |
| Brushless (30 RPM, 20 N·m) | 1 | TBD | TBD | Brushless durability bakeoff |
| **Hip Yaw — Faradyi** ($365.90 total) | | | | |
| 30 RPM brushed planetary (1:130) | 1 | $51.20 | $51.20 | Compact-frame brushed alternate |
| 60 RPM brushed planetary (1:71) | 1 | $68.40 | $68.40 | Compact-frame fast variant |
| 30 RPM brushless planetary | 1 | $62.90 | $62.90 | Compact-frame brushless alternate |
| L-brackets + M6 screws | 3 + 16 | — | $16.40 | Mounting hardware |
| DHL freight to Royal Oak | — | — | $167.00 | Shipped 30-day lead |
| **Linear Actuators** | | | | |
| Yuhuang YH8-523D | 12 | $53 | $636 | Primary knee build, validated lifetime baseline |
| Yuhuang YH8-520P | 1 | $160 | $160 | Ball-screw architecture evaluation |
| Sunline SLA08 (500N) | 6 | $84 | $504 | Knee speed comparison (high no-load speed) |
| Sunline SLA08 (2000N) | 6 | $84 | $504 | Hip-pitch load-margin primary |

**Bench evaluation criteria:**
1. Load-speed curve under controlled axial load (validate datasheet claims)
2. Continuous current draw at rated load (thermal characterization)
3. Encoder pulses per output unit (closed-loop control resolution)
4. Accelerated wear cycling at representative duty cycle (lifetime validation)
5. Acoustic signature and vibration profile
6. Gearbox material teardown on 1 destructive sample (Yuhuang YH8-523D #13)

---

## Open Questions / Follow-ups

- [ ] Confirm Sunline pricing: $84 for both 500N and 2000N variants in writing (verify quote isn't an error)
- [ ] Sunline lifetime number under application duty load (request engineering review like Simon's)
- [ ] Sunline gearbox material (steel-on-steel, steel-on-bronze, or polymer)
- [ ] JST VHR-6N termination availability and MOQ at all linear actuator suppliers
- [ ] Yuhuang gearbox material confirmation
- [ ] Yuhuang 4.5A figure: stall current or continuous at 500N load?
- [ ] Static load analysis on krab leg geometry to validate hip-pitch peak force assumption (~1,500N)
- [ ] Xingli pricing and lead time on the 8-unit hip yaw order
- [ ] Bench rig design for parallel actuator wear testing (fixture for 4–6 units simultaneously under controlled load)
- [ ] Clevis pin order from Bolt Depot: 50× M12 stainless clevis pins + 100× M3 cotter pins for actuator joints

---

## Mechanical Interface Standards

**Clevis pin standard:** M12 across all linear actuators (matches existing 12mm shoulder bolt standard at leg pivot bearings). Sunline Y02 connector code (⌀12 hole), Yuhuang YH8-523D 12.2mm connector option. Hardware sourced from Bolt Depot (~$1/pin) rather than McMaster (~$10/pin).

**Exception:** If Aigwell 300W is reactivated as a sample unit, its 14mm minimum clevis requires either an adapter tab on the leg-link fixture or a parallel 14mm pin SKU. Not currently planned.

---

## Document Index

| File | Description |
|---|---|
| [`Zhejiang_Xingli_-_Z5D120-24GU-30S-5GU100KB-Model.pdf`](./Zhejiang_Xingli_-_Z5D120-24GU-30S-5GU100KB-Model.pdf) | Xingli 90mm 120W brushed gearmotor (representative; production unit varies by ratio) |
| [`Ningbo_-_DC_GEAR_MOTOR_CATALOGUE_10.pdf`](./Ningbo_-_DC_GEAR_MOTOR_CATALOGUE_10.pdf) | Ningbo Leison full DC gear motor catalog (reference / not ordered) |
| [`faradyi-X60R68118S-71K.png`](./faradyi-X60R68118S-71K.png) | Faradyi X60R68118S-71K customer drawing (1:71 ratio, 16 N·m, 60 RPM) |
| [`faradyi-X60R63126S-130K.png`](./faradyi-X60R63126S-130K.png) | Faradyi X60R63126S-130K customer drawing (1:130 ratio, 20 N·m, 30 RPM) |
| [`PI-FDTX20260421_Faradyi.pdf`](./PI-FDTX20260421_Faradyi.pdf) | Faradyi proforma invoice — 3 motors + brackets + DHL freight, $365.90 |
| [`YH8-523D_Linear_Actuator_Specification.pdf`](./YH8-523D_Linear_Actuator_Specification.pdf) | Yuhuang YH8-523D trapezoidal-screw linear actuator (knee primary) |
| [`YH8-520P_Linear_Actuator_Specification.pdf`](./YH8-520P_Linear_Actuator_Specification.pdf) | Yuhuang YH8-520P ball-screw linear actuator (architecture sample) |
| [`The_detailed_specification_of_SLA08_linear_actuator.pdf`](./The_detailed_specification_of_SLA08_linear_actuator.pdf) | Sunline SLA08 linear actuator full selection manual (500N + 2000N variants) |
