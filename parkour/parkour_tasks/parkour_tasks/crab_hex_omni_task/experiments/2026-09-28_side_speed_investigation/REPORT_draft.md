# Why omni side walking is slow: synthesis (2026-09-28, read-only)

**Status: DIAGNOSIS, not causal proof. Nothing was trained or evaluated.** Four analyst lenses were checked by four
adversarial verifiers, and one new CPU check was added (the onset time course below).

**Verdict.**
- **The reward is not what makes omni side walking slow, and neither is the plant.**
  - W's own reward weights prefer the sideways head's fast lateral gait by about **+0.40/s** in steady state.
  - The same plant walks sideways at 0.18–0.225 m/s in 12/12 sideways-task heads.
- **What caps omni side speed is a gait attractor the policy falls into: "lean and shuffle".**
  - The body shifts about 0.25 m toward the travel direction over its planted feet in the first 2–4 s of every side hold.
  - That parks the trailing row outboard and the leading row folded inboard.
  - From there each stance sweeps only 0.02–0.03 m relative to the body.
  - The feet then advance at about **0.095–0.10 m/s**, and that is the whole steady side speed.
- **What makes omni training form and keep this attractor is not yet known.** It is some factor held constant across all
  103 omni runs. Co-training with fwd/back and an omni-config constant are the two candidates; one side-only omni run
  from scratch (about 1.6 GPU-h) separates them.

## Scale of the gap

| | Side speed at 0.22 | Steady feet-centroid speed | Lean (m) | Stance sweep (m) | Trailing / leading foot vs femur (m) |
|---|---|---|---|---|---|
| W (omni head of record) | 0.10 | 0.095–0.100 | 0.25 | 0.024–0.029 | +0.27 / −0.20 |
| t3 X08 (clock live, from scratch) | 0.10 | 0.092–0.100 | 0.24 | 0.022 | +0.26 / −0.19 |
| Sideways head (creep at 0.25) | 0.22–0.25 | 0.21–0.25 | ≈ 0 | 0.10–0.11 | ≈ 0 / ≈ 0 |
| W fwd (at 0.25) | 0.20 | 0.20–0.22 | — | — | — |

The sideways head is commanded 0.25 m/s, not 0.22, so this comparison is **not speed-matched**.

## New CPU check: the lean forms first, then the feet plateau

Scripts: `/tmp/claude-1000/sidespeed/syn/win_centroid.py` and `win_posture.py`. Both read raw npz in 2-s windows from
the start of the hold. "Lean build" is body speed minus feet-centroid speed. All speeds are in m/s.

| Window | W body | W feet | W lean build | t3 feet | Sideways head feet |
|---|---|---|---|---|---|
| 0–2 s | 0.139 | **0.037** | +0.099 | 0.046 | 0.100 |
| 2–4 s | 0.111 | 0.084 | +0.026 | 0.085 | 0.207 |
| 4–6 s | 0.097 | 0.095 | 0.000 | 0.092 | 0.234 |
| 6–8 s | 0.098 | 0.100 | −0.001 | 0.097 | 0.246 |
| 8–10 s | 0.093 | 0.096 | −0.003 | 0.100 | 0.247 |

- **W's "fast start" is the body shifting over its feet, not a better gait.** W's side speed appears to decay within a
  hold (0.125 → 0.095), but that is the lean forming. The feet themselves accelerate from 0.04 to about 0.10 and stay
  there.
- **The attractor is closed-loop.** It re-forms in every hold, identically at t3 (clock live) and W (clock-free).
- **The sideways head never leans.** Its feet reach 0.21 m/s within 4 s.

## Ranked causes

| # | Cause | Confidence | Status |
|---|---|---|---|
| 1 | **Lean-and-shuffle attractor** caps side speed: both rows sit at the stance end of their stroke, sweep 0.02–0.03 m, and feet advance at about 0.10 | medium | **Primary, proximate.** The description reproduces in every omni stamp and in no sideways stamp. No intervention has tested causation. |
| 2 | **The omni training regime forms and locks the attractor.** A formation/optimisation failure, not incentive. | medium ("not incentive"), low (which factor) | **Primary, root class.** The factor is unidentified. |
| 2a | Co-training with fwd/back in one network: the side gait is built on the forward gait's wide stance | low | Untested; see GPU-1 |
| 2b | An omni-config constant: clock product vs sum, xy tracking kernel (std 0.2, L1 −2) vs travel kernel (std 0.316, L1 −1), facing reward and relay, 4-image mirror, obs layout, RSI speed clamp, lateral band b = 0.25 | low | Untested; see GPU-1, then GPU-2a |
| 3 | Plant lateral ceiling of about 0.22 m/s (runner clip, hip slew or policy; not separated) | medium | Contributor to the ceiling, not to the deficit |
| 4 | Ladder stages (about −0.01 m/s), reward-floor clip (11 % vs 7 % of marginal income), taps on the leading row | low | Minor, or symptoms of #1 |

**Why "not incentive".**
- **Steady-state pricing.** Priced with the sideways head's startup transient removed, W's economy pays its lateral
  gait about +0.40/s more:
  - tracking +0.23;
  - regularisers about +0.2;
  - foot_idle −0.02.
- **The local margin is positive.** Another +0.05 m/s of side speed pays +0.10/s, yet the policy stays at 0.10.
- **Three reward campaigns landed their doses and moved nothing.**
  - Each bought taps, not stride.
  - At formation, the centring bonus moved the lean the wrong way (+0.022 m).
- **RSI already shows the fast gait.** It seeds the fast gait's states in about 10 % of resets, and the gait is not retained.

**Why "factor unidentified".**
- Side speed is invariant to everything varied so far: 0.07–0.124 m/s across about 94 walking heads (sd 0.011).
- That covers every curriculum, clock live and hidden, 3k–12k iterations, and seeds 2 and 3.
- The lean appears in side-only 1b (forked from fwd 1a) and in from-scratch t3 (co-trained). Neither run isolates
  co-training from the omni config.

## Analyst claims after verification

| Claim | Verdict | Why |
|---|---|---|
| E1 flat reward landscape (+0.075/s) | **refuted** | The sideways head's first 3 s are a startup transient after a stand with ML/MR lifted. Steady state gives +0.40/s. |
| E2 jerky-gait kernel flattening | **refuted** | In steady state the fast gait is no jerkier (sd 0.133 vs 0.14–0.15). |
| E3 foot_idle cancels 2/3 of the advantage | **refuted** | −0.02/s in steady state. The telemetry is counter carry-over across resamples. |
| E4 no forward-era or regulariser brake | holds | Endpoint comparison; the slopes are underpowered. |
| E5 reward-floor clip | holds (minor) | 11 % vs 7 % of marginal income |
| CMD-1 fixed at formation, curriculum-invariant | weakened | The fwd–side gap is already present at most t3/1c heads. The 1b vs sideways-1a comparison is confounded. |
| CMD-2 / H3 band b = 0.25 | weakened | Back shares the cap and tracks at 0.79. The sideways heads plateau at 0.22 even at 0.35. Untested. |
| CMD-3 asymmetric command encoding | weakened | The encoding is symmetric (slots 8/9/10). Co-training itself is untested. |
| CMD-4 clock cadence law | **refuted** | Clock-free W is still slow. |
| K1 posture as limiter | weakened → **rank 1** | The posture reproduces. Knee pinning is not necessary for it (t3 flips the knee and keeps the posture). Causation untested. |
| K2 exploration starvation on the knee | **refuted** | The sideways head is equally saturated. |
| K3 runner clip sets a 0.22 ceiling | weakened | The ceiling is real; clip vs slew vs policy is not separated. |
| K4 taps from hip slew | weakened | W fwd has the same slew and a normal apex. |
| K5 slip as drag | weakened | rho > 1 mostly restates the small sweep. |
| H1 formation/search failure, not plant or incentive | weakened, core kept | The sideways formation used a sum clock, the travel kernel and L1 −1, so it is "not the same economy". |
| H2 multi-direction interference | weakened | Correlational. 1b and t3 both lean. |
| H4 obs and facing | weakened | The relay is barely active on side. |
| H5 C1d cost | weakened | About −0.01 m/s, about 1.6× seed noise, not specific to C1d |

## Ruled out

| Candidate | Killed by |
|---|---|
| Reward prices slow side walking; flat landscape | +0.40/s steady-state preference for the fast gait; positive local margin |
| Forward-era terms on side rows | `_cmd_gate` is command-general. L1 pays for speed. The others have weight 0. |
| foot_idle as a brake | −0.02/s in steady state; the telemetry is carry-over |
| Regulariser tax on speed | The endpoint gaits favour the fast gait on every term except cross_vel and foot_idle (−0.03/s together) |
| Clock law or clock reward | t3 and W equally slow. The product clock pays the fast gait more. |
| Under-training side | Side has the most dwell. Speed is flat from 3k to 12k iterations. |
| Facing relay, command encoding | The relay is active on 7–12 % of side steps. The encoding is symmetric. |
| Exploration saturation (omni-specific) | The sideways head is equally saturated |
| Plant, joint limits, actuator caps, A15+B mounting | Same plant at 0.18–0.225 in 12/12 sideways heads |
| 1a fork as the cause of W's deficit | t3 and 1c_004 are from scratch and just as slow |
| Lean sets the stride | Within omni the lean varies at a flat stride; both rows sit at their stroke ends regardless |
| W side speed decays within a hold | The decay is lean formation; the feet accelerate and then plateau |

## Experiments (nothing has run)

| # | Test | Splits | Cost | Confirms / kills |
|---|---|---|---|---|
| CPU-1 | Onset time course (`win_centroid.py`) on 1a, 1b, 1c_003, 1c_004, 2b and the 14 arms | #1 as attractor; hint for 2a | ≈ 20 min CPU | Every side-trained stamp builds about 0.2 m of lean in 2–4 s and then plateaus at 0.07–0.12 → #1 holds. 1a leaning on lat_l → forward-template hint (2a). |
| GPU-0 | Eval-only side sweep at 0.15, 0.22, 0.30 and 0.35 on W and the sideways head | #1 vs band calibration; ceiling #3 | ≈ 0.3 GPU-h | W feet speed flat at about 0.10 with lean about 0.25 → attractor cap. W climbing toward 0.18 at 0.30–0.35 → band story revived. Sideways head flat at about 0.22 → ceiling confirmed. |
| **GPU-1** | **Side-only omni formation from scratch**: pool stand 0.2 / side_l 0.4 / side_r 0.4, t3 economy, 3000 iterations; controls are t3 and B_f s2/s3 plus sideways 1a | **2a vs 2b** | ≈ 1.6 GPU-h + 0.2 eval | ≥ 0.17 m/s with lean < 0.05, sweep ≥ 0.08 and progress ≥ 0.075 at iteration 1000 → **co-training (2a)**. ≤ 0.12 with lean ≥ 0.13 → **omni-config constant (2b)**. |
| GPU-2a | If GPU-1 is slow: two bundled side-only arms, **R** (sum clock, travel kernel, L1 −1, facing off) and **C** (fore-aft mirror, hidden vy/wz/facing, RSI off, b = 0.35) | which half of 2b | ≈ 3.4 GPU-h, then ≈ 1.6 GPU-h per split | The arm that reaches ≥ 0.17 with no lean holds the factor |
| GPU-2b | If GPU-1 is fast: W side-only fine-tune for 4000 iterations against B s3/s2 | interference vs formation lock-in | ≈ 2 GPU-h | Lean < 0.1 and feet ≥ 0.15 → interference still acts, so fix with side-weighted stages. Unchanged → locked at formation, so fix with a side-first stage or a warm start from the sideways head. |
| GPU-3 (optional) | W resume with `KRABBY_OMNI_PROGRESS_W` 2.0 against B | whether income can buy the transition | ≈ 2 GPU-h | Expect a TIE plus more taps. A falling lean would reopen the reward route. |

The recommended order is CPU-1 → GPU-0 → GPU-1, about 2 GPU-h before the first branch point. GPU-2a and GPU-2b are
conditional on GPU-1.

## Caveats

- **The evidence is observational.** Posture and slowness co-occur across the omni/sideways split, and that split also
  differs in lineage, reward, commands and observations.
- **The sideways-head comparisons are not speed-matched.**
- **The omni stamps are not independent.** Most are re-evaluations inside a few lineages.
- **The +0.40/s pricing uses deterministic eval rollouts, not stochastic training rollouts.** It also leaves out the
  stride and yaw terms, which are small.
- **GPU-1 and GPU-2 are single-seed.** Add a second seed only for a result within 0.03 m/s of a boundary.
- **Scratch scripts.** All under `/tmp/claude-1000/sidespeed/`:
  - analysts: `econ.py`, `kernel.py`, `kin3.py`, `arms.py`, `explore.py`, `hist/*`;
  - verifiers: `v_ss.py`, `v_win*.py`, `v_air.py`, `verK/*`;
  - synthesis: `syn/win_centroid.py`, `syn/win_posture.py`.
