# FB6k + INC0: B confirmed at fresh seeds on the FB pool (results)

2026-10-04. Pre-registered in [CHARTER.md §11](../CHARTER.md) and reviewed before launch. The reading comes from
`scripts/fb6k_read.py` (`fb6k_read.{json,md}`); the records are in `gate/`. An independent fable·high check
([CHECK.md](CHECK.md)) re-derived every number from the raw files and **PASSES**.

## The answer: the FB6k gate is PASS, and INC0 is 0/3

From scratch at the confirm seeds 16–18, on the FB pool (stand / fwd / side_l / side_r / back at 0.2 each), the B
package forms **both side hands by 6000 iterations in 3/3 seeds**. The C0 lineage stays frozen in 0/3. B also
improves forward, back and the untrained diagonal.

| | B s16 | B s17 | B s18 | C0 s16 | C0 s17 | C0 s18 |
|---|---|---|---|---|---|---|
| side L / R (m/s) | FAST 0.203 / FAST 0.202 | FAST 0.203 / FAST 0.198 | FAST 0.226 / FAST 0.187 | SLOW 0.096 / 0.111 | SLOW 0.106 / 0.085 | SLOW 0.101 / 0.098 |
| suite fwd | 0.996 | 0.933 | 0.968 | 0.691 | 0.579 | 0.645 |
| suite back | 0.956 | 0.917 | 0.933 | 0.715 | 0.605 | 0.687 |
| suite side | 0.856 | 0.859 | 0.874 | 0.437 | 0.443 | 0.450 |
| suite diag (untrained) | 0.917 | 0.822 | 0.964 | 0.546 | 0.483 | 0.582 |
| fall share (side holds) | 0.000 | 0.000 | 0.000 | 0.020 | 0.025 | 0.013 |
| median \|μ\| cam / rod | 0.54 / 0.88 | 0.52 / 0.88 | 0.51 / 0.84 | 7.59 / 9.15 | 6.84 / 9.22 | 8.34 / 9.42 |
| side jitter | 0.152 | 0.149 | 0.156 | 0.096 | 0.104 | 0.108 |

- **Formation timing:** each B run's second hand crossed between model_3000 and model_4500, or by 5999 (s18). No
  run needed the 9000-iteration extension. Halving each hand's pool weight (C_lr 0.4 → FB 0.2) cost about 1.4× the
  iterations, not 2×: formation is sub-linear in weight. A from-scratch continuous run reproduces x6k's fork result
  in kind.
- **Forward and back are not hurt; they improve.** The seed means are 0.966 / 0.935 against C0's 0.638 / 0.669,
  and B's suite survival is 1.00. The check shows C0's low ratios are real weaving walks, not falls: only 69 % of
  C0's 0.5-s path windows are within 20° of the command, against 96–100 % for B.
- **diag is genuine diagonal travel,** with 91–100 % of path windows within 20° of the command and only a mild
  8–10° facing drift. It is descriptive only, because diag is untrained here.
- **INC0** is C0 with only `action_rate 0`, on C_lr. It is FAST/FAST in 0/3, but each run formed **one** FAST hand,
  where C0 formed 0 of 16. Its jitter is 0.36–0.60 with fft 2.9–4.2.
  - **What this implies:** removing the action-rate penalty is what lets the first hand form. The construction
    (bound loss + soft σ 0.8 + entropy 0.0025) adds the **second hand**, and it cuts jitter from 0.36–0.60 down
    to 0.15.
  - **Caveats:** INC0 ran on C_lr, not the FB pool, and s13's second hand was still rising at 6000.

## What the PASS does and does not license (independent check)

It licenses that **the pre-registered precondition for P1 is met.** B forms both side hands from scratch at
fresh seeds on a multi-command pool, while C0 freezes.

It does not license:
1. **Significance beyond the count rule:** 3/3 against 0/3 gives a two-sided Fisher p of 0.10. The margins are
   wide (B's second hands are +0.022 to +0.042 m/s over the FAST line), and B's 5/5 in x6k agrees.
2. **Smoothness:** B's jitter is 1.4–1.6× C0's at every seed. B's fft equals C0's (1.42–1.49 against 1.40–1.57),
   so this looks like a larger stroke rather than chatter. That is unmeasured against any real-robot bar.
3. **Clock-free walking:** every run here had the clock visible and gait locked to it. Whether the two-hand
   stroke survives the c6x clock cut is untested.
4. **Separation from the priv_reg ramp:** every second-hand crossing happened during the 0.037 → 0.091 ramp. C0 is
   flat under the same ramp, so the ramp is unlikely to be the sole cause.
5. **Robustness to unseen walk+turn:** on the untrained move_turn holds, B's survival collapses to 0.00–0.48,
   against C0's 0.00–0.82. That matters for P1, where turning is trained.

## Decision for you: P1 (the full phase-1 chain, which reads diagonals, turning and the clock cut)

The design is [p1_design.md](p1_design.md). Nothing launches without you.

| # | Option | What it settles | GPU-h |
|---|---|---|---|
| 1 | **P1-B6:** the phase-1 chain (t3 → c4 → c5 → c6 → c6x, clock hidden), B lineage against a C0 lineage at seeds 16–18. t3 is re-sized to 6000 from FB6k's timing (15,000 its total). A pre-registered stop after t3 saves ≈ 27 GPU-h if formation fails once turning shares the pool. A c6 read before the cut separates "never formed" from "lost at the clock cut". The final gate is on the c6x heads, clock hidden: side FAST/FAST plus every category (fwd, back, diag, turn, move_turn) against the C0 lineage, with W as a head-to-head reference. | Your real question: two-hand side walking in the pipeline, clock-free, with **diagonals and turning trained** | ≈ 45 (≈ 17 if the t3 stop fires) |
| 2 | **Clock-cut probe:** fork the three FB6k B heads into the c6x pool with the clock hidden for 4000 its, W as reference. | The cheapest read on P1's two biggest risks: the clock cut, and sharing the net with turn, move_turn and diag. A diagnostic, not a pipeline candidate. | ≈ 6 |
| 3 | **Smoothness probe:** fork the three FB6k B heads with `action_rate −0.3` restored for 3000 its. | Whether a formed two-hand stroke survives restored action smoothing. A precondition for any bake, and independent of P1. | ≈ 4.5 |

- **Free CPU read first:** the high-band (> 3 Hz) action-energy fraction of B against C0 from the saved raws.
  It says whether B's extra jitter is stroke or chatter.
- **Recommendation:** P1-B6. If you want the clock-cut answer before committing ≈ 45 GPU-h, run option 2 first.
  Smoothness is handled after P1 as a pre-registered S-stage (≈ 3 GPU-h), or by option 3 now.
- **Still open from step 3:** decisions 1–2 in [RESULTS.md](../RESULTS.md) (guard reading; MQ proposal).
