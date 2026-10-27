# Turn-band diagnostics (CHARTER 13.2): evals only, 2026-10-05

- **Reader:** `scripts/sf_read.py --no-tb` (full table in `turndiag_read.md` / `.json`).
- **Row:** `turn_band__omni`, from the campaign-local `scenarios_turn_band.yaml`: 100 episodes, an 18-s episode,
  a 12-s hold, deterministic.
- **The read:** in-band means ±0.5 rad, inside P1's trained 0.2–0.6 rad facing band. The ±1.0 rad row is 0.4 rad
  beyond the band and is descriptive.
- **Column key:** `heading_final` is the median final |heading error| in rad (the facing controller's dead band is
  0.133 rad); fc is facing_closed and yp is yaw_progress.

| head | in-band heading_final | in-band fc | in-band yp | 1.0 rad heading_final | 1.0 rad fc |
|---|---|---|---|---|---|
| B t3 s16 / s17 / s18 | 0.225 / 0.181 / 0.161 (mean 0.189) | 0.55 / 0.64 / 0.68 | 0.021 / 0.026 / 0.028 | 0.95 / 0.85 / 0.74 | 0.06 / 0.15 / 0.26 |
| C0 t3 s16 / s17 / s18 | 0.102 / 0.082 / 0.111 (mean 0.098) | 0.80 / 0.83 / 0.78 | 0.040 / 0.046 / 0.040 | 0.40 / 0.19 / 0.30 | 0.61 / 0.81 / 0.70 |
| FB6k B s16 / s17 / s18 (no turn training) | 0.310 / 0.345 / 0.217 | 0.38 / 0.32 / 0.56 | 0.016 / 0.013 / 0.023 | 0.46 / 0.84 / 0.71 | 0.54 / 0.16 / 0.30 |
| W | 0.071 | 0.86 | 0.064 | 0.079 | 0.92 |

**Sampled actions** (policy σ, suite row, diagnostic only), 90° turn fc deterministic → sampled:

| head | deterministic | sampled |
|---|---|---|
| B t3 s16 | 0.04 | 0.11 |
| B t3 s17 | 0.11 | 0.16 |
| B t3 s18 | 0.17 | 0.20 |
| C0 t3 s16 | 0.39 | 0.38 |

Side, fwd and back ratios barely move under sampling (B side 0.56–0.71 sampled vs 0.60–0.79 deterministic).

## Reading (descriptive; nothing is gated here)

1. **B's turn deficit is not only out-of-band. This corrects `p1/DIAGNOSIS.md`.**
   - At the trained 0.5 rad offset, B turns at about 0.6× C0's yaw rate.
   - B ends 0.19 rad off against C0's 0.10, which is C0 + 0.09 rad. That is outside the pre-registered +0.05 rad
     "in-band settles" margin, and above the 0.133 rad dead band, so it is not resting error inside the dead band.
   - The diagnosis line "B turns in training (yaw_progress ≥ C0's)" does not carry over to deterministic evals.
     In-band, C0 closes faster. The training statistic is measured differently: sampled actions, pooled with the
     other categories, and including envs already at their target. It is not a like-for-like read.
2. **Out of band, B's deficit is much larger.**
   - At 1.0 rad, B closes 0.06–0.26 against C0's 0.61–0.81. B mostly stops after the first burst, as the 90° hold
     showed.
   - The diagnosis's out-of-band finding stands, as the larger part of the gap.
3. **Turn training helped B, but not to C0's level.** The FB6k B heads, never trained to turn, end 0.22–0.35 rad off
   in-band. After 6000 P1 t3 iterations, B is at 0.16–0.23.
4. **Exploration is not the limiter.** Sampling B's own noise moves its 90° fc by +0.03 to +0.07. C0 is unchanged.
   A deterministic mean stuck in a local optimum would show a larger sampled gain.
5. **What W does that neither t3 lineage does:** W is at 0.07–0.08 rad in-band and out of band alike. W has had the
   full 12000-iteration chain (C1d turn-first), so this is a head-vs-head reference only.

## Consequences under the pre-registered rules (no rule changes)

- **The t3x rule** needs the B t3x heads' in-band heading_final mean ≤ 0.098 + 0.05 = 0.148 rad for TAILS
  CANDIDATE. The B t3 heads sit at 0.189.
- **The final gate's COST (turn, out-of-band) relabel** applies only if B's c6x in-band turn is within +0.05 rad of
  the C0 lineage's. At t3 it is not.
- **The queue continues** (t3x started 09:59).
