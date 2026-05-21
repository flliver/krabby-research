# Krabby-Uno — Bill of Materials

Rough mechanical + electrical parts list for the **Krabby-Uno** hexapod-style robot. Quantities follow a **six-leg** layout unless noted.

> **Costs:** Dollar amounts in the companion spreadsheet are **planning estimates** (typical retail, Alibaba MOQ-style, or PCB fab ballparks). Update them against real quotes before purchasing.

---

## 1. Leg structure (×6 legs)

Each leg uses three segments: **hip**, **femur**, and **tibia**.

| Segment | Nominal length | Material options |
|--------|----------------|------------------|
| Tibia | 32″ | 2″×2″ dimensional lumber **or** ~13/16″×5″ plywood (rounded trapezoid) |
| Femur | 27″ | 2″×4″ dimensional lumber **or** ~13/16″×5″ plywood (rounded rectangle) |
| Hip | 24″ | 2″×4″ dimensional lumber **or** ~13/16″×5″ plywood (rounded rectangle) |

---

## 2. Motors & actuation

| Qty | Item | Notes |
|-----|------|--------|
| 18 | Linear actuator, **24 V** brushed ~30 W, ~20 mm/s, **5-wire**, linear/slide **pot** feedback | **3× per leg** (six legs). Reference family: mini linear actuators with 10 kΩ pot (e.g. Alibaba). |
| 6 | Brushed **~60 W** motor, **5–6 wire**, hip **yaw** (not a linear actuator) | **1× per leg.** Drives **2× spur gears** for yaw; pair with joint hardware in §3. |
| 6 | Spur gear sets + mounting (hip yaw) | **Detail TBD** — mate to yaw motor above. |

**Linear actuator — design / sourcing notes:** Upgrade candidates include heavier medical/lift-style **24 V DC** linear actuators (e.g. double medical-bed types). Expect a **speed vs. torque** tradeoff: **~80 mm/s** on yaw may be unrealistic with the largest options; hip/knee might stay nearer **~40 mm/s** while you tune balance (socket/horn on knee/hip).

---

## 3. Fasteners, bearings & hinges

| Qty | Item | Notes |
|-----|------|--------|
| 2 | 4″ door hinges | [Home Depot — Design House 4″ square corner, 10-pack](https://www.homedepot.com/p/Design-House-4-in-Square-Corner-Satin-Nickel-Door-Hinge-Value-Pack-10-per-Pack-181669/302034830) |
| 1 | Knee & hip joint kit (bearings, shoulder bolts, washers, spacers, Belleville, nuts — **design TBD**) | Stack serves tibia/femur/hip pivots; see joint options below. |

**Joint design options (knee & hip):**

- **Design A:** ~12 mm × ~3″ shoulder bolt → washer → **6201** bearing (press-fit and/or epoxy into plywood) → 13/16″ plywood → **3/4″ spacer** (4″ HDPE discs or 13/16″ plywood, waxed) → 13/16″ plywood → washer → Belleville washer → lock nut.  
- **Design B:** Same spacer stack as A, but a **surface-mounted thrust bearing** instead of a press-fit **6201**. Concern: downward frictional load on the carriage bolt; **6201 (Design A) preferred** for now.

---

## 4. Wiring & connectors

| Qty | Item | Notes |
|-----|------|--------|
| 1 | JST **XA** 4-position housing (with latch) | Pot / feedback to H-bridge board — [Mouser XAP-04V-1](https://www.mouser.com/ProductDetail/JST-Commercial/XAP-04V-1?qs=uQD7XCvsSCP0bv02UpEBHw%3D%3D&countryCode=US&currencyCode=USD). Crimp tool: see XA contacts row. |
| 4 | JST XA crimp contacts | [Mouser SXA-001T-P0.6](https://www.mouser.com/ProductDetail/JST-Commercial/SXA-001T-P0.6?qs=QpmGXVUTftHsF93uqbkooQ%3D%3D&countryCode=US&currencyCode=USD). **Tool:** [JST-style crimper](https://www.amazon.com/dp/B078WPT5M1) (required for XA contacts). |
| 1 | Molex Micro-Fit **2-pin** housing | [43025-0200](https://www.mouser.com/ProductDetail/Molex/43025-0200?qs=4XSMV6Twtb2B3B7qTwcqgQ%3D%3D&countryCode=US&currencyCode=USD) |
| 2 | Molex Micro-Fit terminals | e.g. cut-strip reel. **Tools:** Dupont-style crimper works; [alternative Micro-Fit–style crimper](https://www.amazon.com/dp/B0FJ8LCZ9W). Official Molex hand tools are expensive — use what matches your pin family. |
| 1 | Molex **Mega-Fit** 2-pin housing | [170001-0102](https://www.mouser.com/ProductDetail/Molex/170001-0102?qs=MxnWX8BLHKcasIkfhonQpw%3D%3D&countryCode=US&currencyCode=USD) |
| 2 | Mega-Fit terminals | [76823-0322](https://www.mouser.com/ProductDetail/Molex/76823-0322-Cut-Strip). **Tool:** trial [Mega-Fit crimper](https://www.amazon.com/dp/B0BJKLLCRM); official Molex bench tools are very costly — consider simpler blade/ring terminals long-term if crimps are painful. |

---

## 5. Cage / body

### Frame panels

| Qty | Part | Material / notes |
|-----|------|------------------|
| 1 | Bottom (and/or top) | 3/8″ plywood **28″×48″**; use 1/2″ on top only if needed to reduce flex when standing |
| 2 | Short sides | ~10″×28″ — strong enough for hip mounts (**13/16″** plywood and/or dimensional lumber) |
| 2 | Long sides | ~10″×48″ — 3/8″ plywood or other light material |
| 8 | Diagonal braces | Triangles — plywood or dimensional lumber; brace walls and tie lid to bottom |
| 4 | Lateral braces | **24″** — battery support + lid support |
| 2 | Battery dividers | **15″** lateral braces between battery bays |

### Fasteners (wood screws)

| Qty | Screw | Use |
|-----|-------|-----|
| 3×16 | 1.5″ | Diagonal braces → sides / front / back |
| 2×8 | 1.5″ | Lateral braces → front / back |
| 4×1 | 1.5″ | Lateral braces → battery compartment braces |
| 24×1 | 1.5″ | Bottom → sides / front / back |
| — | Hinge + latch + support | Top lid — mechanism TBD |
| 4×1 | 1″ | Power terminal block |
| 2×6 | 1″ | H-bridge power board chassis |
| 2×3 | 1″ | Leg-pair control boards |

---

## 6. Power distribution

| Qty | Item | Notes |
|-----|------|--------|
| 2 | **12 V 100 Ah** LiFePO₄ (or similar) batteries | [Example listing](https://www.amazon.com/dp/B0DGT9YR7D?ref_=ppx_hzsearch_conn_dt_b_fed_asin_title_5) — wired **in series** for 24 V |
| 1 | **150 A** distribution block, **6×** 1/4″ posts | Battery bus to motors — [example](https://www.amazon.com/maierke-Distribution-Automotive-Terminal-Terminals/dp/B0D3PMJ412) |
| ~6 ft | **10 AWG** tinned copper marine wire (2-conductor) | To H-bridge boards — [example](https://www.amazon.com/dp/B0D796TF4Z?ref_=ppx_hzsearch_conn_dt_b_fed_asin_title_1) |
| 12 | **10 AWG** ring/lug, 1/4″ stud | Motor end from H-bridge — [example](https://www.amazon.com/DTLDZSC-Electric-12-10Awg-1-Terminals-Connectors/dp/B0CYWSM78V) |
| ~8 ft | **2 AWG** stranded copper (≈4 ft red + 4 ft black) | Battery → battery → distribution — [example](https://www.amazon.com/Gauge-Battery-Power-Cable-Black/dp/B0CQN62NQP) |
| 8 | **2 AWG** lugs, 1/4″ | Battery series + distribution |
| 1 | **150 A** inline circuit breaker / fuse holder | [Example](https://www.amazon.com/Nilight-Resettable-Overload-Protection-Amplifier/dp/B0CFY6J6DW) — may need to remove inner core for 2 AWG fit; place on **+** from 2nd battery to block, **within ~7″** of battery |

---

## 7. Controller electronics

| Qty | Item | Notes |
|-----|------|--------|
| 3 | **ATmega2560** boards (Arduino Mega–class) | e.g. [Alibaba Mega2560 R3 / CH340G](https://www.alibaba.com/product-detail/High-Quality-MEGA2560-R3-CH340G-Open_62260090293.html?spm=a2756.trade-list-buyer.0.0.602076e90nyL4v) |
| 3 | **Krabby-Uno** Mega cable-routing shield (custom PCB) | Design not published yet |
| 3 | 3D-printed cases (Mega + shield) | **TODO** — snap lid; mounting for shield; cutouts for 2× DB-25, USB, barrel, 2× 3-pin JST serial, power LED |
| 6 | **Krabby-Uno H-bridge** power boards (custom) | — |
| 6 | 3D-printed H-bridge enclosures | **TODO** — snap lid; M* mounting; cutouts for ~10 LEDs, 1× DB-25, 3× Micro-Fit, 3× 4-pin JST, 1× Mega-Fit; rear area for **heatsinking** (TBD) |
