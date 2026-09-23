# M1 post-mortem: the policy mean is saturated (bang-bang control)

*2026-09-23. The ladder is parked at `await_user:M2` after D05. The user chose "stop and rethink σ" and asked for a
plan to fine-tune M1. CPU-only analysis; no GPU was used.*

## D05: M1 was a clear REJECT

M1 reset σ 2.0 → 0.4 with a clamp at 0.5 (X03, fork of X08:c6, 2000 it).

**Q got WORSE:** z −1.36 against R 0.32 (seed-dominated, from D00bs).

| metric | change |
|---|---|
| action jitter | +10–15 % in every category |
| turn swings per rad | 63 → 79 |
| move_turn swings per rad | 101 → 132 |
| move_turn speed-ratio guard | 0.504 → 0.433 (WORSE) |
| back slip-per-metre guard | WORSE |
| stand six-down fraction | 0.003 (unchanged) |

**σ followed the review's prediction:** 0.40 at model_8000, then 0.50 at model_9000 and model_9995, pinned at the clamp.

**Training itself degraded.** Late-window values, compared with X01 (the knob-free control fork):

| | X03 (M1) | X01 (control) |
|---|---|---|
| tracking income | 0.61 | 0.69 |
| episode length | 1629 | 1758 (more falls under M1) |

Tracking and episode length were still falling at the end of the window rather than recovering.

## Root cause: the deterministic mean sits far outside the ±1 clip

Measured on the stored raw policy output (`actions` in the suite npz is pre-clip):

| head | frac \|μ\| > 1 | frac \|μ\| > 2 | median \|μ\| | p90 \|μ\| |
|---|---|---|---|---|
| X08:c6 (reeval) | 0.960 | 0.912 | 8.1 | 16.2 |
| X01 (X0 s3) | 0.964 | 0.922 | 9.1 | 18.1 |
| X03 (M1) | 0.965 | 0.925 | 9.5 | 19.0 |

Every one of the 18 dimensions is saturated 94–98 % of the time. The executed action is effectively sign(μ): a bang-bang
command to ±0.25 rad (rods) or ±π rad/s (cams), rate-limited only by the actuator slew.

| X08:c6 reeval, first 400 episodes | frac \|μ\| > 1 | sign flips per dim per s | median \|μ\| |
|---|---|---|---|
| zero command (stand / settle) | 0.956 | **2.23** | 7.4 |
| moving | 0.965 | 1.83 | 8.5 |

**Why the legs move at stand.** A bang-bang controller cannot hold an intermediate joint position, so it chatters. The
command flips sign more often at stand than while walking. This matches stand `six_down_frac` 0.004 and a stand contact
switch rate of 1.65 Hz, equal to walking.

**The mechanism.**
- At σ = 2.0 under a ±1 clip, the only way to execute a chosen action reliably is |μ| ≫ σ.
- PPO therefore drives the mean far past the clip. This is the clipped-Gaussian bang-bang pathology covered by the lit
  review §6 (Seyde et al.; the clip bias of Chou et al. and Fujita & Maeda; Andrychowicz et al. on tanh vs clipping).
- Cutting σ afterwards cannot pull μ back, because nothing penalises |μ| beyond the clip. M1's μ even grew
  (median 9.1 → 9.5).
- Lower σ removed the noise that had been "averaging" the command. The mean's own switching then reached the actuators
  more directly, which fits M1's rise in jitter.

## Implication

Plan cause 1 is re-stated. The problem is the saturated mean, not σ as such. The candidate fix is a loss-level
action-bound term in the PPO update that penalises |μ| beyond about 1. IsaacGymEnvs / rl_games carry a
`bounds_loss_coef` for this; that precedent is from memory and must be verified before it is cited. The fix would be
paired with a σ cap the mean can live inside. It is to be planned as an M1 fine-tune (user, 2026-09-23).
