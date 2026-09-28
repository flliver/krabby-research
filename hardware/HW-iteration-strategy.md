# Hardware Electronics Dev Strategy

Krabby Co | July 2026 | Fletcher, for review with Nick

## Why this matters

Board iteration is our slowest and most expensive learning loop. Software revs in hours, mechanical revs in days on the AVID CNC at i3, but a board rev through JLCPCB takes 1.5-2 months end to end: design, review, fab, ship, assembly, debug. Every electrical design mistake costs a full cycle and roughly $3k in design labor plus fab.

The v0.2 boards showed the real cost of that loop. We shipped the MCU and power boards to fab, and only afterward established that the BTS7960 half-bridges sit above their 28V absolute max on a full LiFePO4 charge (~29.2V). That is a part-selection error that a same-week hardware validation loop would have caught before fab money and two months went out the door. Instead the fix is rolling into v0.3.

And v0.3 raises the stakes. We are moving from one centralized ATmega + shield stack to six identical per-leg controller boards: STM32G474, 3x STSPIN9P2x full bridges, CAN-FD daisy chain to the Orin, flash-over-CAN. That is a denser, mixed-signal board (72V-rated power stage next to small-signal analog and a 5 Mbps bus) with more ways to be subtly wrong, and it is the design that carries into the fleet. After that comes the BLDC drive stage migration. Rev count goes up, not down, and each rev on the current loop is two months.

## Phase 1 (now): contract design + JLCPCB fab

- v0.3 leg board RFQ is out to Oleksandr and Mairaj. Winner designs against our spec; JLCPCB fabs and assembles.
- Design files live in our GitHub repo; schematic and layout reviews are checkpointed before anything goes to fab, which is our main defense against another v0.2-class miss.
- This remains the path for all production boards regardless of what follows. Multilayer, plated vias, and assembled 30A power stages need a real fab.

## Phase 2 (trigger-based): in-house rapid prototyping, $10-50k capex

The goal is to compress validation revs from weeks to same-day by printing prototype boards in-house, so design errors surface on the bench before fab, not after.

**Voltera discussion summary.** We evaluated Voltera's desktop systems for exactly this. After a technical call, their rep ruled out their entry V-One for our application: our fine pitches (64-pin STM32G474, STSPIN packages) are at the edge of what it reliably produces and would take significant trial and error. He steered us to their NOVA dispensing system (~$46k), which handles our feature sizes on rigid substrates with silver inks, other metal inks, epoxies, and adhesives. He also flagged their Alta pick-and-place, pre-launch, where a $100 reservation locks 30% off the purchase price. Wider market context from earlier research: true 4-layer desktop PCB printing does not exist at reasonable cost, so NOVA-class single/double-sided printing is the realistic ceiling for in-house.

**Honest scope limit.** Printed silver-ink boards validate logic, signal routing, CAN, and sensor chains. They do not carry motor current or replace plated through-hole power stages. What this buys is fast iteration on the MCU/signal side of our boards, which is where most of the v0.3 novelty and risk lives. Power sections still validate through the fab path.

**Purchase triggers, any one of:**

- SBIR Phase I award or equivalent funding lands
- We are running 3+ board revs per quarter and contract turnaround is the blocking dependency
- A hire comes on who will own the tool day to day

## Phase 3 (vision): closed-loop AI hardware iteration

This is the same pattern as everything else we build: commodity components orchestrated by software, with the value in the orchestration layer. Plywood legs, vacant lots, open-source stacks; boards are next. AI proposes design revs against test results, the in-house printer turns them into hardware within a day, the bench harness validates, and only converged designs go to JLCPCB for production quantities. Design labor shifts from drafting to review, and the two-month fab cycle becomes a production step instead of the learning loop.

## Ask

No spend now. Agreement on the Phase 2 triggers, and a green light to place the $100 Alta reservation.