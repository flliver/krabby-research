# Literature review: a separate stand network -- DAgger of a scripted re-stance expert, action-history copycat, discontinuous targets and policy handover -- mapped to the Krabby campaigns

*2026-10-11. Prepared as step 0 of the stand-network campaign
(`crab_hex_omni_task/experiments/2026-10-11_stand_network/PLAN.md`, decisions D1–D6, audited 3 passes) after the
p2s and p2r stand campaigns closed without a bake (`experiments/2026-10-07_omni_phase2/P2R_CLOSE.md`; CHARTER.md
§12, L1170–1955) and while the step-2 gate calibration (`calib.md`, 15/15 cells land) is complete and no training has
run. Builds on `lit-review-gait-motion-quality.md` (stand-still rewards, Isaac Lab standing envs),
`lit-review-rl-algorithm-and-ppo-hyperparameters.md` §13 (distillation targets, rsl_rl Distillation, Parkour in the
Wild fine-tuning), `lit-review-omni-reward-simplification.md` and `lit-review-omnidirectional-gait-formation.md`,
which it does not repeat. Every citation was opened by the review's verification pass and checked against the
paper's abstract or full text, or the raw source file, at time of writing; six load-bearing ones were re-opened at
writing time ([3, 17, 20, 36, 43, 52]). Where verification corrected a sweep claim the corrected wording is used,
and items that failed verification or are only inferred are in the appendix rather than softened.*

## 1. The problems this review targets

Restated from P2R_CLOSE.md [83] (its lessons 1–11) and PLAN.md / calib.md [84] so this document stands alone:

1. **March in place.** The phase-2 head of record S1 s18 (`policy/6_p2a_field/model_19994.pt`) completes the wall
   field at 0.8125 (field_hard, n = 400) but at the latched stand slot steps at 2.83 steps/s with 0.135 m lifts; on the
   G1 STILL clauses it fails steps 2.833 (line 0.5), clearance 0.135 (0.05), off-ground 0.338 (0.10), taps 2.83 (1)
   and foot-speed KS 1.72 (0.5) at both env seeds [84].
2. **In-network stand income costs walking.** p2s put foot-support and slide terms into the shared network and
   field_hard fell 0.8125 → 0.6175; the same specialist routed only at the stand slot (ST2M) kept the walls (0.835)
   but still tapped 16.1 / 15.3 per s (lesson 1) [83, 84].
3. **Statues without homing.** Z1 (statue prior, stand bit) reached 0 / 0.083 steps/s, field_hard 0.83, 0 / 1500
   stand falls, but 55 % five-foot stands with cams unhomed (hip yaw 4.9–8.7°); HOLD (zero action) likewise: homed
   0 %, G2 FAIL, yaw ≈ 6° (lesson 10) [83, 84].
4. **Offline distillation of the scripted re-stance controller R failed closed loop.** A statue prior toward R (Z2)
   had no effect (hip bias −0.13…−0.18 open-loop on clean HOLD rows). Clean-state BC on R-labelled banks (Z2c) passed
   the open-loop gate (hip 0.020 / cam 0.016) yet ran at 50.9 / 47.7 taps/s, homed 0 %, off the manifold within 3–4
   steps. Masking last action + proxy then fine-tuning (Z2d) was STILL but homed 1.7 / 2.3 % (settled 17–19 %). The
   LIFT onset (hip 0 → −0.5 when |cam| crosses eps_step 0.3) never fit: μ_hip −0.054 on LIFT rows, onset rows
   0.22–0.25 % of the banks; the HOLD cam-trim slope fit −0.04…−0.12 against R's −1 because clean rows sit at
   |c| ≈ 0.0035; hip bias −0.046 with the proxy masked, −0.088 with last action masked, +0.003 with both: the network
   read its phase from its own last action (lessons 3–6) [83].
5. **Routing a separate controller is free.** R at the latched stand slot while S1 walks (`P2R_RSTg_s18`) homes
   97.0 / 97.7 % of walkstop stands, settle median 2.92 / 2.96 s, STILL, 0 stand falls, field_hard 0.825 (n = 400),
   field_turn 1.0, walkstop onset 0.12 s (REFS 0.14, HOLD 0.12); the hard switch (entry on the first latched-stand
   step, no debounce, exit on the step the slot changes) gave 0 handover falls (lesson 11) [83, 84].
6. **The plan.** D1: an S1-shaped `CrabHexActorCriticRMA` warm-started from S1, estimator frozen, actor-side mask of
   the own-action / current-sense columns 65–88 as arm `A-mask` vs unmasked `A-full`. D2: frozen S1 in the training
   loop, stand pool ≈ 0.5, push DR ±0.5 m/s every 8 s, per-episode DART σ ∈ {0, 0.1, 0.3}. D3: DAgger imitation of R,
   per-episode β → 0 over the first third, a fixed 10 % of episodes R-executed, capped reservoir buffer, MSE on the
   actor mean, minibatches balanced by R branch (HOLD / WAIT / LIFT / HOME / SEAT) with LIFT-onset rows oversampled.
   D5: hard switch, no blending. D6: closed-loop gates only (G1 STILL, G2 homed ≥ 90 %, G3 falls incl. exit-handover,
   G4 walls non-inferior, G5 onset). ≈ 12 GPU-h; RL only as a later fallback [84].

**The short answer.** The field has two answers to stand/walk interference, one network with mode-routed rewards or
a separate skill; Krabby has paid for the first (p2s routed its stand income to the slot and still lost 0.19 of
walls) and measured the second at zero walking cost, so D1 stands (§2). Seeding the receiving skill from rollouts of
the frozen outgoing policy has one published precedent, Go2-W, and it is D2; a hard switch chosen by a layer trained
with the real low-level policies in the loop is reported smooth on hardware, the documented failures being hand-timed
switches (§3). Z2c/Z2d are the textbook offline-BC failure, but DAgger's gain is conditional: R must recover from the
*student's* failure states, which the 97 % probe does not measure, and a one-state error at an unrecoverable onset
still compounds as T² (§4). State-gated takeover with hysteresis and a quantile budget is better supported than a
fixed 10 % R-executed slice (§5). The copycat mechanism is the Z2d diagnosis exactly; on-policy relabelling is its
principled fix and the mask only a partial one, so both arms stay (§6). The LIFT onset is the conditional-mean ramp an
MSE head cannot represent with ideal data; R's free branch label makes a branch-gated head the cheapest structural
fix, which the plan does not carry (§7). Noise must be sized to the measured closed-loop error with labels kept clean
(§8); RL on top, conservative and regularised toward R, or a residual over R, is the fallback (§9). R's sequenced,
interlocked re-stance is classical hexapod and insect practice; homing is a plant requirement, not a biological one
(§10).

## 2. One network or two: reward interference, mode routing and stepping at zero command

**Key papers:** Peng et al. 2025 [4]; Haarnoja et al. 2024 [1]; Lee, Hwangbo, Hutter 2019 [2]; Matsuzawa et al.
2026 [3]; van Marum et al. 2024 [5]; legged_gym [68, 69]; Isaac Lab velocity and Spot [70, 71]; OmniH2O [48];
HWC-Loco [10]; Kim et al. [72]; Portela et al. [73]; CaT [74].

Peng et al.'s no-routing ablation "activates mutually conflicting gait-specific rewards at every timestep (e.g.,
stillness from standing vs. forward velocity from walking/running)": mean episode length 11.59 against 972.6 with
gait-ID reward routing (31.26 without curriculum), all gaits including Stand in one LSTM policy [4]. Haarnoja et al.
found end-to-end training without separate soccer and get-up skills converged "to either a poor locomotion local
optimum of rolling on the ground or focusing on standing upright and failing to learn to score", then distilled the
separate skills into one agent [1]. Lee et al. trained three behaviours and a selector individually, which "made it
easier to design well-defined RL tasks and troubleshoot on the real robot"; their multi-task policy showed "frequent
slippages and highly conservative postures" [2]. Go2-W switches a PPO stance policy beside flat- and rough-terrain
policies "according to commands from the higher-level layer" [3]. van Marum et al.: "in many cases" RL walkers have no
native standing mode and walk in place; where standing is supported "a separate standing controller is usually
designed or trained"; switching is hard because the state spaces have "little native overlap", which is why they chose
one stand-and-walk controller [5].

Stepping at zero command is the default of the reference recipes: legged_gym gates feet-air-time off at |cmd| < 0.1
and ships `stand_still` at weight −0. [68, 69]; Isaac Lab's base velocity env has no stand term
(`stand_still_joint_deviation_l1`, threshold 0.06, is defined and unused), `rel_standing_envs` 0.02, `last_action` in
the obs [70]; its Spot config swaps in a stance reward clip(contact_time − air_time, ±0.3) and scales the always-on
joint-position penalty ×5 once the command is zero and body speed ≤ 0.5 m/s, foot slip always on, 10 % standing envs
[71]. OmniH2O: "the humanoid tends to take small adjustment steps instead of standing still" (fixed by data variants, a
max-feet-height reward and a curriculum) [48]; HWC-Loco: "Some even exhibit in-place stepping at 0 m/s" [10]. Flag
fixes: a stand-mode indicator below 0.2 that keeps all legs in contact and zeroes the cyclic functions [72]; a
contact schedule zeroed below 0.1 m/s [73]. Stillness is not homing: requiring double contact on walk-to-stand makes a
policy "opt for the closest stance position rather than the most stable one" [5]; CaT's c_still = (‖q − q*‖ − ε)·
1[v_des = 0] forces "the robot to go back to its default pose" [74].

**What it means here.** Both answers separate the stand objective from walking; p2s was the routed-single-network
route and lost 0.19 of walls, Z1 and HOLD are van Marum's closest-stance failure, and the routed controller cost
nothing [83]: D1 is the Go2-W / Lee route with the cost already measured. The latched slot does structurally what the
flag fixes do in-network; G2 homed ≥ 90 % must stay a gate separate from G1 STILL [5, 74].

## 3. Handover: start states, the switch, and the exit bridge

**Key papers:** Go2-W [3]; Lee 2019 [2]; ABS [6]; van Marum [5]; Barkour [7]; Lee et al. CoRL 2021 [8]; ANYmal
Parkour [9]; HWC-Loco [10]; Reske et al. [11]; MELA [12]; Peng [4]; Tidd et al. [13]; Lee et al. ICLR 2019 [14];
Christmann et al. [15]; Li et al. (Cassie) [16].

Diagnosis: "jerky motions when the previous specialist policy reaches a state that is unfamiliar to the following
policy" [7]; "a naive composition of skills fails when a policy encounters a starting state never seen during its
training" [8]. Fixes are several: (a) widen the receiving skill's start states, which only Go2-W does by rolling out
the frozen outgoing policy ("a separately trained locomotion policy driven by random commands is also used to generate
mid-locomotion states as initial configurations") [3], ABS by hand-set v_x ~ U(−0.5, 5.5) m/s and roll/pitch
U(−π/6, π/6) [6], Lee 2019 from each behaviour's own start distribution with the behaviours "regarded as a part of the
environment" [2]; (b) regularise the outgoing skill's terminal states, since widening start sets "quickly becomes
infeasible as the larger initiation set often leads to an even larger termination set", so "keep the termination set
small and near the initiation set of the next policy" [8], and ABS's recovery posture term targets "a nominal standing
pose with low height allowing the robot to switch back to the agile policy seamlessly" [6]; (c) cycle-time constraints
or output blending [5]; (d) one distilled or unified policy [7, 5]. The switch: ANYmal Parkour freezes the skills,
deploys the selector's argmax, reports switches "smooth and unnoticeable on the real robot" and says rolling out the
real low-level policies in training "is necessary" [9]; Lee's learned selector "smoothly switches ... without any
noticeable delay" where a hand-designed FSM was "often unsmooth" and switched "at a bad timing and falls" [2]. ABS
(value threshold −0.05) and HWC-Loco (one-hot selector, switch penalty) hard-switch and say nothing about smoothness;
HWC-Loco lists discrete switching over fixed controllers as a limitation [10]. Switching separately trained policies
"can cause jerky and unstable locomotion" unless "one undesirably uses the stance mode for transitioning" [11]; MELA
co-trains all experts with its gate [12]; Peng et al. enter Stand only after low speed and double support persist
1.5 s ("This hysteresis prevents premature switching") [4]. One opened paper's learned blend of frozen skills beat a
*manual* hard switch on hardware (appendix); no reviewed paper compares a learned blend with a learned-selector hard
switch. Exit bridges: walk → jump switching 1.5 % direct vs 82.2 % with a setup policy, 12.3 without the extended reward
(post-handover reward credited to the bridge), 38.8 without initialising the bridge from the walking policy, 0.0 for a
learned when-to-switch gate alone [13]; transition policies with proximity predictors [14]; a transition-net success
classifier gating at 0.95, +19 % on the hardest pairs [15]. Caution: standing-only Cassie policies "regardless of
whether they were trained with external perturbations, the robot keeps losing its balance if it is forced to lean
beyond its support region", while a walk+stand policy steps to recover [16].

**What it means here.** D2 is the Go2-W precedent and the only one; D2's push DR is ABS's widening. D5 matches the
ETH evidence and Krabby's own (0 handover falls, onset 0.12 s), and entering the stand *is* the benign stance-mode
transition [11, 83]. The exit is the exposed side: R's SEAT target is the terminal-state regulariser of [8, 6], so
release the slot only from it; if G3's exit clause fails, credit S1's first post-exit steps to the stand net (Tidd's
extended reward) before adding a bridge. The Cassie result says a stand net with no stepping recovery needs a
push-perturbed read and an explicit exit to S1 under large disturbance. Peng's hysteresis is a borrow for R's WAIT
branch only if exit falls appear; the no-debounce latch has worked.

## 4. Offline BC versus DAgger: bounds, recoverability and the legged recipe

**Key papers:** Ross, Gordon, Bagnell [17]; Pan et al. [18]; Simchowitz et al. [19]; Rajaraman et al. 2020, 2021
[20, 21]; Swamy et al. 2021 [22]; Spencer et al. [23]; IFM [24]; Khadiv et al. [25]; Lee et al. 2020 [26]; Miki et
al. [27]; RMA [28]; Parkour in the Wild [29]; Robot Parkour Learning [30]; rsl_rl Distillation [31]; MPC-Net [32].

Supervised imitation on expert states: J(π) ≤ J(π*) + T²ε, tight; DAgger: J ≤ J* + uTε + O(1), linear in T only
when the recoverability constant u = O(1); Super Tux Kart supervised imitation did not improve with more data while
DAgger "never falls off the track after 15 iterations"; the Mario expert was a near-optimal planner with full state
[17]. Pan et al. require an expert that can "perform stably under arbitrary action perturbation, regardless of where it
starts", show the batch gap "grows quadratically in T", and saw batch IL crash in a region "rarely explored in the
expert's demonstrations" [18]. With continuous actions and no interaction, any smooth, deterministic or
simply-stochastic Markovian imitator can suffer execution error exponentially larger in the horizon than its training
error, even with a smooth expert and exponentially stable dynamics; the escapes are non-smooth, non-Markovian (chunked,
diffusion) or state-dependent-stochastic policies, a "spread" expert distribution, or interaction [19]. Interaction is
not a guarantee: the ≳ |S|H²/N lower bound holds "even if ... the learner is allowed to actively query the expert", so
DAgger-type algorithms "cannot improve over behavior cloning in the worst case" [20]; under μ-recoverability interaction
achieves Õ(μ|S|H/N) against ≳ |S|H²/N [21]; on-Q matching is O(T) for O(1)-recoverable problems and O(T²) for
O(T)-recoverable ones, and "a classification error of κ on a single state can lead to Ω(κT²) imitation gap" even with
interactive feedback [22]. Spencer et al.'s regimes: realisable → BC works; non-realisable with partial coverage →
O(T²) "is simply inevitable without repeated access and interaction with the demonstrator"; bounded density ratio → a
simulator correction reaches O(T) [23]. IFM asserts naive BC "often fails" and used < 400K DAgger samples, not ablated
[24]; counterpoint: Khadiv et al. learned trot, jump and bound by purely offline BC with states perturbed in the
contact-constraint nullspace, failing only with torque action spaces and end-to-end architectures [25].

The legged recipe is on-policy from the start, β = 0: "training data is generated by rolling out trajectories by the
student policy" (TCN student, batch 20,000, 4,000 iterations) [26]; "rolling out the student policy to increase
robustness", student initialised from teacher weights "to speed up training" [27]; expert rollouts "contain examples of
only good trajectories", so the randomly initialised adaptation module trains on-policy [28]; a_student + N(0, σ²)
executed with the expert queried each step [29]; expectation over the student's rollouts [30]; rsl_rl's Distillation
executes the student's stochastic output, labels with `teacher(obs)` and `storage.clear()`s every update, with no β
mixture and no aggregation [31]. In the theory β_i weights a mixture that "queries the expert to choose controls a
fraction of the time", only its mean must go to 0, and β_i = I(i = 1) "often performs best in practice", though in
Mario 0.5^(i−1) slightly beat it (3030 vs 2980) and 0.9 was slower [17]; Pan ran 3 iterations at expert probabilities
60 / 36 / 21 % [18]; MPC-Net ramps the learner's share linearly 0 → 1 [32].

**What it means here.** Z2c is the T²ε signature (open-loop PASS, 50 taps/s closed loop, off-manifold in 3–4 steps),
and a discontinuous R with a smooth MLP is the non-realisable regime where interaction is required [23]. The gain is
conditional on recoverability [20–22]: the probe's 97 % / 2.9 s measures R from S1's handover states, not from the
student's failure states (five-foot stands, hover at |c| ≈ eps_step, unhomed cams), and a missed LIFT onset is a
single-state error that compounds for the rest of the stand [22]; pre-check R's takeover from those states before
step 7's premise stop. D3's per-episode β over the first third is slower than every legged reference and R-executed
episodes are the clean data Z2c showed does not transfer: use the indicator or a 0.5^i decay. D3's reservoir buffer is
the right departure from rsl_rl since onset rows are 0.22–0.25 %; warm-starting from S1 has Miki's precedent.

## 5. State-gated DAgger: when the expert takes over

**Key papers:** SafeDAgger [33]; HG-DAgger [34]; EnsembleDAgger [35]; LazyDAgger [36]; ThriftyDAgger [37].

SafeDAgger learns a classifier for ‖π − π*‖² > τ, executes the expert when it fires and aggregates only those states;
it reduces to DAgger with β_i = 0 for large τ, learned faster than DAgger, and the expert's share fell faster [33].
HG-DAgger keeps the expert in control until the system is back in a "permitted set" and adds only those recovery
segments; it ties DAgger's late-epoch instability to the β decay [34]. EnsembleDAgger lets the novice act only if
discrepancy < τ and ensemble variance < χ, assuming a "primarily unimodal" expert [35]. LazyDAgger switches to the
supervisor on predicted discrepancy > β_H and back only when the ground-truth discrepancy < β_R < β_H (a hysteresis
band) and injects isotropic Gaussian noise into the supervisor's executed actions; removing either lowered MuJoCo
performance, with ~60 % fewer context switches than SafeDAgger [36]. ThriftyDAgger intervenes on novelty (ensemble
variance > δ_h) or risk (1 − Q̂ > β_h), thresholds set as (1 − α_h) quantiles for an intervention budget α_h, cedes
only when ‖π_r − π_h‖² < δ_r and risk < β_r, and beat HG-, Lazy- and SafeDAgger on peg insertion with each ablation
significantly worse [37].

**What it means here.** R is free, so every visited state is labelled regardless; gating decides which states are
*executed and recovered from*. A discrepancy-gated takeover (R executes when the per-channel |μ − R| exceeds β_H, hands
back only after a full seat cycle below β_R) targets the stuck states (hover above eps_step, unhomed cam) and is a
drop-in for D3's fixed 10 % slice: set the budget by quantile [37], keep the hysteresis so a mid-LIFT leg is not handed
back [36], and log the takeover rate beside step 7's homed share. The evidence is driving and manipulation with human
experts; EnsembleDAgger's unimodal assumption is strained at LIFT.

## 6. The action history: copycat, inertia and the mask

**Key papers:** de Haan et al. [38]; Wen et al. 2020 [39]; Codevilla et al. 2019 [40]; Swamy et al. 2022 [41];
Spencer [23]; Ortega et al. [42]; Keyframe-Focused IL [43]; ChauffeurNet [44]; Chuang et al. [45]; Yang et al. (DIR)
[46]; Lee et al. 2020 [26]; HOVER [47]; OmniH2O [48]; IFM [24].

With the previous action in the observation rather than random noise, "access to more information can yield worse
performance"; DAgger needed hundreds of queries on MountainCar and tens of thousands on Hopper to undo it [38]. The
imitator can "learn to cheat by predicting the expert's previous action" when expert actions are strongly
autocorrelated and past actions are recoverable from the history; a Walker2D "fails to transition from an extended
position to a bent position", Half-Cheetah "did not ever start to move from a resting position", held-out loss improves
while reward falls as the window grows, and DAgger with 1000 queries approaches the expert [39]. The driving inertia
problem, "spurious correlation between low speed and no acceleration", cannot be separated from true causes "unless an
explicit causal model or on-policy demonstrations are used" and saturates or worsens with more data [40]. Off-policy
methods treat their past actions "as though they came from the expert", a "latching behavior", and "actually perform
worse when given access to history" [41]; the divergence "is simply another manifestation of covariate shift,
exacerbated particularly by settings of feedback" [23]; actions must be treated "as causal interventions" [42].
Offline fixes are not uniformly worse: copycat is loss-optimal when the changepoint fraction is below the
reward-optimal imitator's error (an argument, "not fully rigorous"); weighting rows by the error of an MLP fed only
past actions beat every history baseline on CARLA "including, surprisingly, even DAGGER which has the advantage of 120k
expert queries", but with ego speed observed it only matched single-frame BC, and on MuJoCo DAgger with 1k queries was
marginally better [43]. History dropout is weak: ChauffeurNet drops past motion for 50 % of examples because the net
"can learn to cheat by just extrapolating from the past" [44], yet dropout "performs similarly to BCOH, indicating it
can not solve the copycat problem", while predicting the residual a_t − a_{t−1} beside the current observation beat it
[45]; "removing all information about a_{t−1} may be counterproductive" and single-observation BC "performs poorly, due
to partial observability" [39]. Lee et al. keep previous foot targets in the current observation but out of the TCN
history [26]. For rare continuous targets, reweight by the inverse of a kernel-smoothed label density; synthetic
oversampling "can actually degrade" [46]. DAgger students carrying 25-step action histories (HOVER; OmniH2O, where
DAgger "improves performance overall, especially for policy with history input") and IFM's student with previous
action plus phase report no copycat failure, an absence of report under smooth experts [47, 48, 24].

**What it means here.** Z2d is this signature (held-out fit, phase read from the last action, onset never launched:
Half-Cheetah's rest). On-policy relabelling of the student's own stuck states is the principled fix, so `A-full` is
the conventional recipe; `A-mask` has support only as a partial fix, and Keyframe's full-observability result (R's
branch is a function of roll/pitch, cams and hips, all inside the 75-wide prop view) says it may match `A-full` plus
reweighting. Keep both arms and grade them on μ_hip on LIFT rows as well as G1/G2. An onset fraction of 0.22–0.25 % is
far below any imitator error, so onset weighting is not optional: weight by a last-action-only predictor's error [43]
and smooth over (branch, |c| − eps_step) [46] so the HOLD-slope rows at |c| ≈ 0.0035 are covered. Rows executed by R
(the 10 % slice, the β phase, takeovers) put expert actions back into the history slots: down-weight them in the
student loss or use them only under the mask.

## 7. Discontinuous expert targets: heads, branch gates, chunks and phase

**Key papers:** Bishop [49]; Implicit BC [50]; Behavior Transformers [51]; Conditional IL [52]; hg-DAgger [53];
GLiDE [54]; ACT [55]; Diffusion Policy [56]; MPC-Net [32]; Reske et al. [11]; MoE-Loco [57]; AHC [58]; DeepMimic
[59]; Pfrommer et al. [60]; Tang & Agrawal [61].

Sum-of-squares outputs "approximate the conditional averages of the target data, conditioned on the input vector"
[49]; explicit models "take every intermediate value between training samples", and on a scripted goal-switching
oracle MSE policies hold 95 % success only to 8-D where implicit ones reach 16-D (2,000 demos); on xArm multi-target
pushing MSE 89.7 % vs EBM 99.0 / MDN 99.7 [50]. BeT: an MSE model between two modes "takes 0 action to minimize MSE"; a
k-means bin head with focal loss plus a per-bin offset whose loss is masked to the ground-truth bin; without binning
0.25 (Block push) / 0.68 (Kitchen) relative, a GPT-MDN head 0.30 / 0.83 / 0.86 [51]. CIL: the command "acts as a switch
that selects between specialized sub-modules": branched 88 / 64 % vs the command as an input 78 / 52 %, which "is not
forced to take the commands into account"; without noise-injected recovery data 56 / 22 %, interventions 0.67 → 8.67
[52]. hg-DAgger trains a meta-controller to predict the expert's subgoal label and each subpolicy only on low-level
labels collected while the correct subgoal was chosen, beating flat DAgger and BC with a fraction of the labels
(grid-world, discrete actions) [53]. A mode-conditioned student loses less under perturbation (BC 0.93 / 0.99 / 0.38 →
0.20 / 0.18 / 0.03; mode-gated 0.85 / 0.99 / 0.25 → 0.40 / 0.39 / 0.15) while flat BC is equal or better unperturbed
[54]. Single-step policies on paused demonstrations "can pause indefinitely for certain states"; chunking raised success
from 1 % at k = 1 to 44 % at k = 100 [55]; BC-RNN and IBC "often get stuck" on idle actions [56]. Legged hybrid experts:
MPC-Net's gated mixture switched "precisely at the times when the contact configuration of the system changes" and beat
an equal-latent MLP, with per-leg phase inputs [32]; performance improves when each expert "specializes in controlling
exactly one mode", a guided cross-entropy ties the gate to the observed mode, and augmenting the existing sin(πφ) input
with φ, φ̇ cut constraint violation "by a factor of four" [11]; MoE-Loco's cosine 0.278 vs −0.132 is its most-conflicting
pair, mixed-terrain 0.879 vs 0.571 [57]; AHC's DAgger student is an MoE labelled by a base-height rule [58]; DeepMimic
carries a phase φ ∈ [0, 1] in the state [59]. A log-barrier-smoothed MPC expert beats randomized smoothing [60];
factorised categorical heads stay PPO-compatible [61].

**What it means here.** μ_hip −0.054 on LIFT rows is the conditional-mean ramp [49, 50]: DAgger changes which states
are visited and onset weighting the ramp's slope, but a flat MSE head cannot represent the step with ideal data
[50, 51]. R exposes its branch for free, so a branch classifier (focal or class-balanced cross-entropy) with
branch-masked regression heads [51, 52, 53] is the cheapest structural fix, and its gain should show off-nominal
(handovers, unhomed cams), which is what the closed-loop gates read [54]. It changes D1's S1-shaped checkpoint (a
loader adapter beside `_load_extra`), so it belongs as the pre-registered third arm before RL. Phase without the
last-action channel: R's thresholded quantities (|c| − eps_step, tripod interlock) are state-computable and an
auxiliary branch head on the trunk gives them. Chunking conflicts with a reactive hard switch at the slot; a smoothed
or phase-timed R_s must first pass R's own closed-loop gates as a probe (the plan's later phase).

## 8. Noise: DART and state tubes

**Key papers:** DART [62]; MPC-Net [32]; Khadiv et al. [25]; Swamy et al. ICML 2022 [64]; de Haan [38]; Block et al.
[63]; LazyDAgger [36].

The noise "should approximate the error of the trained robot's policy"; naive isotropic noise "does not perform well,
and leads to unsafe policies during execution"; on Humanoid DART matched DAgger 3× faster, lowering the supervisor's
reward by 5 % where DAgger executed policies with "over 80% less cumulative reward" [62]. MPC-Net samples x ~ N(x_nom,
Σ_x) because a nominal-only policy "is overly aggressive to small deviations" and oscillated on hardware, and still
mixes the learner in because "there is still a bias of the state distribution" [32]; Khadiv et al. perturb at each MPC
re-plan, projected into the contact nullspace [25]. Temporally correlated noise in recorded actions creates spurious
correlations "that a learner might latch on to" [64]; an independent-noise previous-action slot carries no confound
[38]; Gaussian noise on the policy's observation input at training and execution, targets fixed, gives the continuity
property that controls generative-BC error, and noising the learned policy's actions is "not best practice" [63].
LazyDAgger's noise goes on the supervisor's executed actions during interventions, and helps [36].

**What it means here.** Set D2's σ per channel (cam, hip, knee) from the observe pass's closed-loop |μ − R| rather than
{0, 0.1, 0.3}; open-loop residuals were ~0.02, so σ 0.3 on cams under the slew limit may push stands into never-visited
states and degrade both labels and safety [62]. Perturb cams and hips as a state tube before R labels [32, 25]: the
cheap fix for the HOLD trim slope, since clean rows never show the deviations R's −1 slope corrects. Labels stay R's
clean action; the history slot records the executed noised action, which then acts as an intervention (inference from
[38]). Apply the same noise on R-executed takeover rows [36].

## 9. RL on top: conservative fine-tuning, annealed regularisers, residuals

**Key papers:** IFM [24]; Parkour in the Wild [29]; TWIST [65]; Distillation-PPO [66]; AHC [58]; Haarnoja et al. [1];
Residual MPC [67].

IFM: MPC and its DAgger student "fell off 12% of the tests" (never for IFM or vanilla RL); PPO fine-tuning at initial
std 1.0 (2.0 in one setting), lr 1e−5, clip 0.05 and no imitation term improved linear tracking 0.1838 → 0.1335
(vanilla RL 0.2017) and kept symmetric gaits where RL from scratch found "an asymmetric, energy-inefficient gait that is
similar to pronking" [24]. Parkour in the Wild: the DAgger student lost 10.4 % success; "Done naively, the performance of
the policy steadily degrades once the RL loss is applied"; the fixes were action noise in distillation, reduced initial
std with conservative hyper-parameters, and critic pre-training with the policy frozen [29]. TWIST: L = L_RL +
λ D_KL(π_stu ‖ π_tea), λ reduced, "RL+BC >> RL >> BC (DAgger)", because DAgger "can not stably and robustly track unseen
motions occasionally, due to the lack of task reward guidance" [65]; D-PPO 0.5 L2 + 0.5 PPO, qualitative only [66]; AHC
fine-tunes its DAgger MoE on the original task rewards with PCGrad [58]. Haarnoja et al. regularise "to only one of the
two skills" per state, λ rising to 1 while predicted return is below a threshold and falling to 0 above, which let
agents "improve beyond any simple scheduling of the skill policies" [1]. Residual MPC learns "targeted corrections to
the MPC outputs" whose prior "acts as a strong bias, initializing and guiding the policy towards desirable behavior with
a simple set of rewards" [67]. Every expert here is smooth.

**What it means here.** The fallback ladder if DAgger reaches STILL but misses G2 or G3: (1) PPO on the separate net
with stand rewards (support, slide, cam-home, still), reduced initial std, small clip and lr (IFM's values as the start),
critic warm-up with the actor frozen, and an annealed KL/L2 term toward R routed by state (stand slot / R branch),
which is not p2s's failure of learning stand rewards from the walking policy in the shared net; (2) a residual stand
policy a = R + δ that never represents the LIFT step, cannot copycat its way out of it, and keeps R's interlock, the
cheapest arm if the onset cannot be fit.

## 10. Classical and biological re-stance: home or latch, and the interlock

**Key papers:** OpenSHC [75, 76]; Phoenix [77]; MIT Cheetah [78]; Sapkal et al. [79]; Daun-Gruhn et al. [80];
Walknet [81]; neuroWalknet [82].

OpenSHC stops in step with the gait: in STOPPING the velocity input is ignored and the clock runs at zero stride; "All
legs must attempt at least one step to achieve default tip position after ending a swing"; a leg is frozen only at swing
end with zero body velocity at its target tip (a leg that misses accepts its current tip as the new default); STOPPED
only when every leg is at the correct phase and posing is complete; on restart mid-swing legs are forced into STANCE
and MOVING begins only after every leg has completed a first step [75, 76]. Phoenix: "Gait NOT in motion, return to
home position"; a leg lifts to home only in its own slot and only if more than 2 units from home; extra timed cycles run
until all legs are down [77]. MIT Cheetah is the opposite: a Standing gait with all-stance schedule
(0,0,0,0)/(10,10,10,10) that latches the current x/y/yaw and never re-places feet [78]. Drosophila has both: walk-OFF
flies "often took a few steps before halting" and "the leg always assumes a relaxed stance position, regardless of when
the optogenetic stimulus started", while the brake pathway freezes a leg where it is, but a leg stimulated mid-swing
"finishes the swing phase in an unaltered manner, but freezes as soon as it lands on the ground" [79]. A stick-insect
model completes the stance "until the femur reaches its static horizontal position, and the tibia its static flexion"
and restarts in swing or stance by femur angle [80]. Walknet's Rule 1 "inhibits the beginning of swing of the receiver
leg as long as the sender leg is in swing"; its standing circuit "does not require an explicit specification of the
desired leg positions", and the controller converges "to a stable walking pattern when starting from arbitrary leg
configurations" [81]; neuroWalknet's Stand unit sets the global velocity to zero [82].

**What it means here.** R's HOLD/WAIT/LIFT/HOME/SEAT under a tripod interlock is OpenSHC and Phoenix practice and
Cruse's Rule 1; a slot-sequenced re-stance has an onset discontinuity by construction, and the classical stacks time
the lift by gait phase rather than a |cam| threshold, which names a phase-timed R_s as a candidate smoother. Latching
is a legitimate stop in biology and in Cheetah; homing is a plant requirement (cams must be homed for the next start).
If G2 proves unlearnable, "swing-complete then hold" plus a walk that starts from any posture is the fallback, which
HOLD already gives at 0 steps/s and 0 falls. OpenSHC's restart rule is a template for the exit handover and G3's exit
clause. Never cut a leg mid-swing [79].

## 11. Literature gaps

- No paper trains a DAgger student from a scripted, *discontinuous* expert: every legged DAgger teacher is an MPC
  or an RL policy [24–32, 47, 48]; the branch-gated and chunked heads are manipulation, driving or grid-world results
  [51–56].
- No comparison of a learned blend against a learned-selector hard switch; the hard-switch smoothness evidence is
  two ETH hardware reports [2, 9] (appendix).
- Only Go2-W seeds a receiving skill from frozen-policy rollouts, with no ablation of that choice [3].
- State-gated DAgger has no legged or scripted-expert result; its comparisons with β-schedule DAgger are driving and
  manipulation with human experts [33–37].
- No test of the copycat mechanism on legged students; the history-carrying DAgger successes are absence of report
  [47, 48, 24].
- No source softens or hysteresis-relaxes a scripted expert for imitation; smoothing results concern MPC [60].
- DART's calibration rule has no legged instance; executed-action noise in legged distillation is Parkour in the
  Wild only [62, 29].
- No stand-still ablation on a hexapod; the in-network recipes are quadruped/humanoid [68–74]; hexapod stop logic is
  source code and a conference abstract [75, 77, 80].
- Load-bearing 2025–2026 preprints without replication: [3, 4, 10, 19, 29, 57, 58, 65–67].

## 12. Ranked recommendations

Knobs are the plan's (`stand_net` launcher, campaign-local DAgger trainer, `restance_target` constants, `--stand-mask`,
D2 σ, D3 β / 10 % slice / reservoir / branch balance). Cost basis: ≤ 2 GPU-h per arm, ≈ 0.8 per eval head, ≈ 12 GPU-h
planned [84].

0. **Recoverability pre-check, ≈ 0.3 GPU-h, before step 7** (§4; [20, 21, 22, 18]). Route R to take over from
   student-visited failure states (Z2d / Z2c observe-pass states: five-foot stands, hover at |c| > 0.3, unhomed cams)
   and read homed / settle / falls. The 97 % probe measures recovery from S1 handovers only; if R cannot reseat from
   those states, DAgger's bound is quadratic there and the onset needs near-zero error (§7) rather than more on-policy
   data.
1. **Keep D1 (separate net, warm start), D2 (frozen S1 handovers, push DR) and D5 (hard switch)** (§2, §3; [3, 2, 9,
   6, 27]). The campaign's numbers (p2s 0.8125 → 0.6175 in-network; R routed at 0.825 / 97 % / 0 falls / onset 0.12 s)
   agree with the literature; nothing here contradicts D1–D3.
2. **β fast, slice state-gated** (§4, §5; [17, 31, 36, 37]). Replace "β → 0 over the first third" with β_i = I(i = 1) or
   0.5^i; keep a 10 % R-executed control, but spend the executed-R budget on discrepancy-gated takeovers with
   hysteresis (R in when per-channel |μ − R| > β_H, out after a full seat cycle below β_R), budget set by quantile. Log
   the takeover rate as premise telemetry with step 7's homed share. Cost: trainer lines, 0 GPU-h.
3. **Onset weighting by predictor error and smoothed density, both arms** (§6; [43, 46, 39]). Onset rows 0.22–0.25 %
   ≪ any imitator error: weight rows by a last-action-only MLP's error, smooth over (branch, |c| − eps_step) so the
   HOLD-slope rows at |c| ≈ 0.0035 count; down-weight R-executed rows. Grade both arms on μ_hip on LIFT rows (Z2d:
   −0.054) beside G1/G2; `A-mask` is a partial fix in the literature, so the arm pair stays.
4. **σ from the measured residual; state tube before labelling** (§8; [62, 32, 25, 63]). Pick σ per channel from the
   observe pass's closed-loop |μ − R| (open-loop ~0.02) instead of {0, 0.1, 0.3}; perturb cams and hips before R labels;
   labels always clean. 0 GPU-h beyond the arms.
5. **Branch-gated head as the pre-registered third arm, ≤ 2 + 0.8 GPU-h** (§7; [51, 52, 53, 50, 54]). If either arm
   is STILL but misses G2 (Z2d pattern: homed 1.7–2.3 %, LIFT never launched), add a branch classifier on R's free
   label (focal / class-balanced CE) with branch-masked regression heads, or at minimum an auxiliary branch head on the
   trunk; needs a loader adapter beside `_load_extra`. This is where the literature disagrees with the plan: D1's flat
   MSE head is the construction that cannot represent the step, independent of data.
6. **Exit handover** (§3, §10; [8, 6, 13, 75, 16]). Release the slot only from R's SEAT state; if G3's exit clause
   fails, credit S1's first post-exit steps to the stand net (extended reward) and force mid-swing legs to stance at
   exit (OpenSHC's restart) before any bridge policy. Read a push-perturbed stand (D2's ±0.5 m/s DR) since neither R nor
   the net steps to recover.
7. **RL fallback ladder, later phases** (§9; [24, 29, 65, 1, 67]). (a) PPO on the separate net with stand rewards,
   reduced std, clip 0.05, lr 1e−5 (IFM), frozen-actor critic warm-up, annealed KL to R routed by stand slot; (b)
   residual a = R + δ. Only after rung 5.
8. **Hysteresis on entry only if exit falls appear** (§3; [4, 71]); the no-debounce latch has 0 handover falls.

**Where the literature disagrees with the plan.** (i) Separate-vs-unified is a real fork in the field [4, 5]; Krabby's
in-network attempts already route by slot and still lost walls, so the fork is settled here by measurement. (ii) DAgger
is not a guaranteed cure: the worst-case bound equals BC's [20] and a single unrecoverable onset error compounds [22];
step 7's premise stop needs rung 0 first. (iii) The plan treats DAgger + oversampling as sufficient for the onset; IBC
and BeT say the MSE head is the limit with ideal data [50, 51] (rung 5). (iv) A fixed 10 % R-executed slice has no
precedent; budgeted, state-gated takeover does [37]. (v) Homing is a plant requirement; biology and Cheetah latch
[78, 79, 81], so a hold fallback is legitimate.

## 13. References

1. Haarnoja, Moran, Lever, et al. *Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning.* Science Robotics, 10 Apr 2024, DOI 10.1126/scirobotics.adi8022; arXiv:2304.13653. https://arxiv.org/abs/2304.13653 — end-to-end "rolling on the ground or ... standing upright"; get-up skill from key poses; stage-2 per-state regularisation with λ to 1 below Q_s, 0 above.
2. Lee, Hwangbo, Hutter. *Robust Recovery Controller for a Quadrupedal Robot using Deep Reinforcement Learning.* arXiv:1901.07517, 2019. https://arxiv.org/abs/1901.07517 — behaviours + selector trained individually; multi-task "ungraceful behaviors"; behaviours "part of the environment"; FSM "often unsmooth", "bad timing and falls".
3. Matsuzawa, Irie, Yoshida, Suzuki, Hara, Tomono. *Long-Distance Real-World Navigation of the Legged-Wheeled Robot Go2-W Using Deep Reinforcement Learning.* arXiv:2606.21387, 2026. https://arxiv.org/html/2606.21387 — Sec. 4.2 stance policy; three policies switched by high-level commands; mid-locomotion initial states from a separately trained locomotion policy.
4. Peng, Bao, Zhou. *Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion.* arXiv:2505.20619, 2025. https://arxiv.org/html/2505.20619 — episode length 972.6 / 11.59 / 31.26; conflicting rewards every timestep; W2S 1.5 s hysteresis.
5. van Marum, Shrestha, Duan, Dugar, Dao, Fern. *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking.* IROS 2024; arXiv:2404.19173. https://arxiv.org/html/2404.19173 — "in many cases" walk in place; separate standing controller usual; "little native overlap"; Sec. IV-B closest stance.
6. He, Zhang, Xiao, He, Liu, Shi. *Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion.* RSS 2024; arXiv:2401.17583 v2. https://arxiv.org/html/2401.17583v2 — Sec. III-C V_threshold −0.05 switch; Sec. VI-B posture term; Sec. VI-C recovery initial-state DR.
7. Caluwaerts, Iscen, Kew, Yu, Zhang, et al. *Barkour: Benchmarking Animal-level Agility with Quadruped Robots.* arXiv:2305.14654, 2023. https://arxiv.org/abs/2305.14654 — p. 8 jerky transitions at unfamiliar states; BC generalist on 17,636 episodes.
8. Lee, Lim, Anandkumar, Zhu. *Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization.* CoRL 2021; arXiv:2111.07999. https://arxiv.org/abs/2111.07999 — abstract and intro: naive composition fails; larger initiation → larger termination set; keep termination set near the next initiation set.
9. Hoeller, Rudin, Sako, Hutter. *ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots.* Science Robotics 2024; arXiv:2306.14874. https://ar5iv.labs.arxiv.org/html/2306.14874 — frozen inner-loop skills; argmax selection; "smooth and unnoticeable"; rolling out real low-level policies "is necessary".
10. Lin, Qiao, Tai, Li, Jia, Liu. *HWC-Loco: A Hierarchical Whole-Body Control Approach to Robust Humanoid Locomotion.* arXiv:2503.00923, 2025. https://arxiv.org/html/2503.00923 — Sec. 4.3 one-hot selector, switch penalty; Sec. 5.3 in-place stepping at 0 m/s; Limitations on discrete switching.
11. Reske, Carius, Ma, Farshidian, Hutter. *Imitation Learning from MPC for Quadrupedal Multi-Gait Control.* ICRA 2021, DOI 10.1109/ICRA48506.2021.9561444; arXiv:2103.14331. https://arxiv.org/abs/2103.14331 — intro: jerky switching unless stance mode; one expert per mode; guided CE; Sec. V-A phase augmentation ×4.
12. Yang, Yuan, Zhu, Yu, Li. *Multi-expert learning of adaptive legged locomotion.* Science Robotics 5(49), 2020; arXiv:2012.05810. https://arxiv.org/abs/2012.05810 — no wait time from disjointed switching; stage-2 co-training.
13. Tidd, Hudson, Cosgun, Leitner. *Learning Setup Policies: Reliable Transition Between Locomotion Behaviours.* IEEE RA-L 2022; arXiv:2101.09391. https://arxiv.org/pdf/2101.09391 — Table II 1.5 / 51.3 / 82.2 / 0.0; Table I 38.8 / 12.3 / 82.2; extended reward.
14. Lee, Sun, Somasundaram, Hu, Lim. *Composing Complex Skills by Learning Transition Policies.* ICLR 2019. https://youngwoon.github.io/transition/ — transition policies; proximity predictors; bipedal walk/jump/crawl.
15. Christmann, Luo, Soeseno, Chen. *Expanding Versatility of Agile Locomotion through Policy Transitions Using Latent State Representation.* ICRA 2023; arXiv:2306.08224. https://arxiv.org/abs/2306.08224 — transition-net success classifier; 0.95 threshold; +19 % hardest pairs.
16. Li, Peng, Abbeel, Levine, Berseth, Sreenath. *Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control.* IJRR, DOI 10.1177/02783649241285161. https://hybrid-robotics.berkeley.edu/publications/Cassie_RL_Versatile_Locomotion.pdf — case study C: standing-only policies keep losing balance beyond the support region.
17. Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS 2011; arXiv:1011.0686. https://arxiv.org/abs/1011.0686 — Thm 2.1 T²ε; Thm 3.2 uTε; Sec. 3 β schedule, indicator "often performs best"; Sec. 5.1–5.2 Super Tux Kart, Mario (3030 vs 2980).
18. Pan, Cheng, Saigol, Lee, Yan, Theodorou, Boots. *Agile Autonomous Driving using End-to-End Deep Imitation Learning.* RSS 2018; arXiv:1709.07174. https://arxiv.org/abs/1709.07174 — Def. 1 perturbation-stable expert; quadratic batch gap; Sec. IV-D 60/36/21 %; Sec. VI-A boundary crash.
19. Simchowitz, Pfrommer, Jadbabaie. *The Pitfalls of Imitation Learning when Actions are Continuous.* arXiv:2503.09722 v4, 2025. https://arxiv.org/abs/2503.09722 — non-interactive exponential-in-H lower bound; improper-policy and spread-distribution escapes.
20. Rajaraman, Yang, Jiao, Ramchandran. *Toward the Fundamental Limits of Imitation Learning.* NeurIPS 2020; arXiv:2009.05990. https://arxiv.org/abs/2009.05990 — |S|H²/N bound even with active querying; DAgger "cannot improve over behavior cloning in the worst case".
21. Rajaraman, Han, Yang, Liu, Jiao, Ramchandran. *On the Value of Interaction and Function Approximation in Imitation Learning.* NeurIPS 2021. https://proceedings.neurips.cc/paper/2021/hash/09dbc1177211571ef3e1ca961cc39363-Abstract.html — μ-recoverability; Õ(μ|S|H/N) vs ≳ |S|H²/N.
22. Swamy, Choudhury, Bagnell, Wu. *Of Moments and Matching: A Game-Theoretic Framework for Closing the Imitation Gap.* ICML 2021; arXiv:2103.03236. https://arxiv.org/abs/2103.03236 — Sec. 4.5 recoverability; Lemma 7 Ω(κT²).
23. Spencer, Choudhury, Venkatraman, Ziebart, Bagnell. *Feedback in Imitation Learning: The Three Regimes of Covariate Shift.* arXiv:2102.02872, 2021. https://arxiv.org/abs/2102.02872 — previous-action divergence as covariate shift; latching; HARD regime O(T²) without interaction.
24. Youm, Jung, Kim, Hwangbo, Park, Ha. *Imitating and Finetuning Model Predictive Control for Robust and Symmetric Quadrupedal Locomotion.* IEEE RA-L 2023; arXiv:2311.02304. https://arxiv.org/html/2311.02304 — Sec. III-B naive BC "often fails", < 400K DAgger samples, inputs incl. previous action and phase; Sec. III-C fine-tune std 1.0 / lr 1e−5 / clip 0.05; 12 % fall-offs; Table I 0.1838 / 0.1335 / 0.2017.
25. Khadiv, Meduri, Zhu, Righetti, Schölkopf. *Learning Locomotion Skills from MPC in Sensor Space.* L4DC 2023, PMLR 211. https://proceedings.mlr.press/v211/khadiv23a/khadiv23a.pdf — Sec. 3.5 contact-nullspace perturbation at each re-plan; Sec. 4 PD vs torque action spaces; conclusion: no interactive demos needed.
26. Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning Quadrupedal Locomotion over Challenging Terrain.* Science Robotics 5(47), 2020; arXiv:2010.11251. https://arxiv.org/abs/2010.11251 — Methods: DAgger by student rollouts; h_t excludes previous foot targets; Table S8 batch 20,000, 4,000 iterations.
27. Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning robust perceptive locomotion for quadrupedal robots in the wild.* Science Robotics 2022; arXiv:2201.08117. https://ar5iv.labs.arxiv.org/html/2201.08117 — teacher weights initialise the student; student rollouts; noise on height samples.
28. Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* RSS 2021; arXiv:2107.04034. https://ar5iv.labs.arxiv.org/html/2107.04034 — Sec. III-B expert rollouts "only good trajectories"; on-policy adaptation module, randomly initialised.
29. Rudin, He, Aurand, Hutter. *Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-expert Distillation and RL Fine-tuning.* arXiv:2505.11164 v1, 2025. https://arxiv.org/html/2505.11164v1 — Sec. 2.2 student + N(0, σ²); Sec. 2.3 naive RL degrades, three fixes; Sec. 3.1 10.4 % drop.
30. Zhuang, Fu, Wang, Atkeson, Schwertfeger, Finn, Zhao. *Robot Parkour Learning.* CoRL 2023; arXiv:2309.05665. https://arxiv.org/abs/2309.05665 — Sec. 3.2 DAgger distillation over student rollouts; GRU + depth CNN student.
31. leggedrobotics/rsl_rl `rsl_rl/algorithms/distillation.py` (main). https://raw.githubusercontent.com/leggedrobotics/rsl_rl/main/rsl_rl/algorithms/distillation.py — `student(obs, stochastic_output=True)`, `teacher(obs).detach()`, `behavior_loss`, `storage.clear()`.
32. Carius, Farshidian, Hutter. *MPC-Net: A First Principles Guided Policy Search.* IEEE RA-L 2020; arXiv:1909.05197. https://ar5iv.labs.arxiv.org/html/1909.05197 — Sec. II-C N(x_nom, Σ_x); Sec. II-D mixture α 0 → 1, "still a bias"; Sec. III-D nominal-only oscillation; Sec. III-E MoE switching at contact changes.
33. Zhang, Cho. *Query-Efficient Imitation Learning for End-to-End Autonomous Driving (SafeDAgger).* AAAI 2017; arXiv:1605.06450. https://arxiv.org/abs/1605.06450 — Eq. 3 safety classifier; Alg. 1 line 7 subset selection; reduces to DAgger β = 0; Fig. 3.
34. Kelly, Sidrane, Driggs-Campbell, Kochenderfer. *HG-DAgger: Interactive Imitation Learning with Human Experts.* ICRA 2019; arXiv:1810.02890. https://arxiv.org/abs/1810.02890 — Sec. II recovery-only labels, permitted set; Eq. 4 doubt threshold; Sec. IV-A DAgger late-epoch instability.
35. Menda, Driggs-Campbell, Kochenderfer. *EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning.* IROS 2019; arXiv:1807.08364. https://arxiv.org/abs/1807.08364 — discrepancy and doubt rules; "primarily unimodal" expert.
36. Hoque, Balakrishna, Putterman, Luo, Brown, Seita, Thananjeyan, Novoseller, Goldberg. *LazyDAgger: Reducing Context Switching in Interactive Imitation Learning.* IEEE CASE 2021; arXiv:2104.00053. https://arxiv.org/abs/2104.00053 — Sec. V-A hysteresis β_R < β_H; Sec. V-B supervisor action noise; Sec. VI ablations; ~60 % fewer switches.
37. Hoque, Balakrishna, Novoseller, Wilcox, Brown, Goldberg. *ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning.* CoRL 2021; arXiv:2109.08273. https://arxiv.org/abs/2109.08273 — Sec. 4.3 intervene/cede rules; Sec. 4.4 quantile thresholds; Sec. 5.3 peg insertion.
38. de Haan, Jayaraman, Levine. *Causal Confusion in Imitation Learning.* NeurIPS 2019; arXiv:1905.11979. https://ar5iv.labs.arxiv.org/html/1905.11979 — CONFOUNDED vs ORIGINAL; "more information ... worse performance"; App. F DAgger query counts.
39. Wen, Lin, Darrell, Jayaraman, Gao. *Fighting Copycat Agents in Behavioral Cloning from Observation Histories.* NeurIPS 2020; arXiv:2010.14876. https://ar5iv.labs.arxiv.org/html/2010.14876 — copycat conditions; Walker2D / Half-Cheetah; Sec. 5 TCA; Sec. 6 BC-SO, DAgger 1000 queries.
40. Codevilla, Santana, López, Gaidon. *Exploring the Limitations of Behavior Cloning for Autonomous Driving.* ICCV 2019; arXiv:1904.08980. https://ar5iv.labs.arxiv.org/html/1904.08980 — Sec. 3.2 inertia problem; more data saturates or worsens.
41. Swamy, Choudhury, Bagnell, Wu. *Sequence Model Imitation Learning with Unobserved Contexts.* NeurIPS 2022; arXiv:2208.02225. https://arxiv.org/abs/2208.02225 — latching; off-policy worse with history.
42. Ortega et al. *Shaking the Foundations: Delusions in Sequence Models for Interaction and Control.* arXiv:2110.10819, 2021. https://arxiv.org/abs/2110.10819 — actions as causal interventions.
43. Wen, Lin, Qian, Gao, Jayaraman. *Keyframe-Focused Visual Imitation Learning.* ICML 2021; arXiv:2106.06452. https://arxiv.org/abs/2106.06452 — Sec. 4.1 changepoint-fraction argument; Eq. 4 action-prediction-error weights; Sec. 5–6 CARLA vs DAgger 120k queries; with-speed matches BC-SO; MuJoCo DAgger 1k marginally better.
44. Bansal, Krizhevsky, Ogale. *ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst.* RSS 2019; arXiv:1812.03079. https://arxiv.org/abs/1812.03079 — Sec. 4.2 past-motion dropout 50 %.
45. Chuang, Yang, Wen, Gao. *Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction.* ECCV 2022; arXiv:2207.09705. https://arxiv.org/abs/2207.09705 — residual memory module; Sec. 5.3 HD ≈ BCOH; memory-only 0.0; DAgger results.
46. Yang, Zha, Chen, Wang, Katabi. *Delving into Deep Imbalanced Regression.* ICML 2021; arXiv:2102.09554. https://arxiv.org/abs/2102.09554 — LDS inverse-density reweighting; SMOTER/SMOGN can degrade.
47. He et al. *HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots.* arXiv:2410.21229, 2024. https://arxiv.org/html/2410.21229 — DAgger loss; 25-step history incl. actions; 32/32 metrics.
48. He et al. *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning.* CoRL 2024; arXiv:2406.08858. https://arxiv.org/html/2406.08858 — 25-step history; DAgger helps most with history input; "small adjustment steps instead of standing still" and its fix.
49. Bishop. *Mixture Density Networks.* Aston University NCRG/94/004, 1994. https://www.microsoft.com/en-us/research/publication/mixture-density-networks/ — abstract: conditional averages.
50. Florence, Lynch, Zeng, Ramirez, Wahid, Downs, Wong, Lee, Mordatch, Tompson. *Implicit Behavioral Cloning.* CoRL 2021; arXiv:2109.00137. https://arxiv.org/abs/2109.00137 — Sec. 3 discontinuities; Sec. 4 / App. C.5 N-D particle 8-D vs 16-D; Table 3 xArm multi-target.
51. Shafiullah, Cui, Altanzaya, Pinto. *Behavior Transformers: Cloning k modes with one stone.* NeurIPS 2022; arXiv:2206.11251. https://arxiv.org/abs/2206.11251 — Fig. 2 MSE averaging; Sec. 2.3 focal loss; Eq. 2 masked offsets; Table 3 ablations.
52. Codevilla, Müller, López, Koltun, Dosovitskiy. *End-to-end Driving via Conditional Imitation Learning.* ICRA 2018; arXiv:1710.02410. https://arxiv.org/abs/1710.02410 — Sec. IV-A branched vs command input; Table 1 88/64 vs 78/52, no-noise 56/22; Sec. VI-B 0.67 → 8.67.
53. Le, Jiang, Agarwal, Dudík, Yue, Daumé III. *Hierarchical Imitation and Reinforcement Learning.* ICML 2018, PMLR 80; arXiv:1803.00590. https://arxiv.org/abs/1803.00590 — Sec. 4.1–4.2, Alg. 2 hg-DAgger; Sec. 5 / Fig. 2 label efficiency.
54. Wang, Wang, Mao, Hagenow, Shah. *Grounding Language Plans in Demonstrations Through Counterfactual Perturbations (GLiDE).* ICLR 2024; arXiv:2403.17124. https://arxiv.org/abs/2403.17124 — Sec. 3.2, Table 3 perturbation results; pseudo-attractor caveat.
55. Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).* RSS 2023; arXiv:2304.13705. https://arxiv.org/abs/2304.13705 — Sec. V-C pausing; Sec. VI-A 1 % → 44 %; temporal ensembling.
56. Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023 / IJRR; arXiv:2303.04137. https://arxiv.org/abs/2303.04137 — Sec. 4 idle actions, BC-RNN / IBC get stuck; action horizon.
57. Huang, Zhu, Du, Zhao. *MoE-Loco: Mixture of Experts for Multitask Locomotion.* arXiv:2503.08564 v1, 2025. https://arxiv.org/html/2503.08564v1 — Table III cosine 0.278 / −0.132 (one pair); Table I 0.879 / 0.571.
58. Zhao, Wang, Wang, Liu, Lu, Han, Liu, Bai. *Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning (AHC).* arXiv:2511.06371, 2025. https://arxiv.org/html/2511.06371 — DAgger into an MoE; base-height teacher rule; stage-2 task-reward fine-tuning, PCGrad.
59. Peng, Abbeel, Levine, van de Panne. *DeepMimic.* ACM TOG 2018; arXiv:1804.02717. https://arxiv.org/abs/1804.02717 — Sec. 4 phase variable; limitations.
60. Pfrommer, Padmanabhan, Ahn, Umenberger, Marcucci, Mhammedi, Jadbabaie. *On the Sample Complexity of Imitation Learning for Smoothed Model Predictive Control.* CDC 2024; arXiv:2306.01914. https://arxiv.org/abs/2306.01914 — log-barrier smoothed expert; vs randomized smoothing.
61. Tang, Agrawal. *Discretizing Continuous Action Space for On-Policy Optimization.* AAAI 2020; arXiv:1901.10500. https://arxiv.org/abs/1901.10500 — factorised / ordinal categorical heads with PPO/TRPO/ACKTR.
62. Laskey, Lee, Fox, Dragan, Goldberg. *DART: Noise Injection for Robust Imitation Learning.* CoRL 2017; arXiv:1703.09327. https://arxiv.org/abs/1703.09327 — Sec. 1 noise ≈ learner error; Sec. 5.1 parity, 3×, 5 % vs 80 %; isotropic noise unsafe.
63. Block, Jadbabaie, Pfrommer, Simchowitz, Tedrake. *Provable Guarantees for Generative Behavior Cloning.* NeurIPS 2023; arXiv:2307.14619. https://arxiv.org/abs/2307.14619 — execution-time augmentation noise on observations; action noising "not best practice".
64. Swamy, Choudhury, Bagnell, Wu. *Causal Imitation Learning under Temporally Correlated Noise.* ICML 2022, PMLR 162; arXiv:2202.01312. https://proceedings.mlr.press/v162/swamy22a.html — spurious correlations "a learner might latch on to"; IV regression.
65. Ze, Chen, Araújo, Cao, Peng, Wu, Liu. *TWIST: Teleoperated Whole-Body Imitation System.* CoRL 2025; arXiv:2505.02833. https://arxiv.org/html/2505.02833 — L_RL + λ D_KL; "RL+BC >> RL >> BC (DAgger)".
66. Zhang et al. *Distillation-PPO: A Novel Two-Stage Reinforcement Learning Framework for Humanoid Robot Perceptive Locomotion.* arXiv:2503.08299, 2025. https://arxiv.org/html/2503.08299 — α = β = 0.5; Table II qualitative.
67. Jeon, Lee, Hong, Kim. *Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control.* arXiv:2510.12717, 2025. https://arxiv.org/abs/2510.12717 — abstract: targeted corrections; prior as strong bias.
68. leggedrobotics/legged_gym `legged_robot.py` (master). https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot.py — L900–902 `_reward_stand_still`; L891 air-time gate; L351 command zeroing; L218 actions in obs.
69. leggedrobotics/legged_gym `legged_robot_config.py` (master). https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot_config.py — L142 feet_air_time 1.0; L146 stand_still −0.
70. Isaac Lab locomotion velocity `mdp/rewards.py`, `velocity_env_cfg.py` (main). https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py — `stand_still_joint_deviation_l1` threshold 0.06; air-time gate 0.1; rel_standing_envs 0.02; last_action obs; no stand term.
71. Isaac Lab Spot `mdp/rewards.py`, `flat_env_cfg.py` (main). https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py — stance reward clip ±mode_time; stand_still_scale 5.0; velocity_threshold 0.5; foot_slip −0.5; rel_standing_envs 0.1.
72. Kim, Lee, Park. *A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards.* arXiv:2409.15780 v1, 2024. https://arxiv.org/html/2409.15780v1 — Sec. III-D stand-mode indicator < 0.2.
73. Portela, Margolis, Ji, Agrawal. *Learning Force Control for Legged Manipulation.* ICRA 2024; arXiv:2405.01402. https://arxiv.org/html/2405.01402 — contact schedule zeroed below 0.1.
74. Chane-Sane, Leziart, Flayols, Stasse, Souères, Mansard. *CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning.* IROS 2024; arXiv:2403.18765. https://arxiv.org/html/2403.18765 — c_still default-pose constraint.
75. CSIRO Robotics. OpenSHC `syropod_highlevel_controller/src/walk_controller.cpp` (master, accessed 2026-10-11). https://raw.githubusercontent.com/csiro-robotics/syropod_highlevel_controller/master/src/walk_controller.cpp — L560 STOPPING→STOPPED; L584 STARTING FORCE_STANCE; L606 one step to default tip; FORCE_STOP at swing end.
76. Tam, Molnar, Ganz, Steindl, Elfes, Hilliges, Zheng, Shtauch, Kottege. *OpenSHC: A Versatile Multilegged Robot Controller.* IEEE Access 8, 2020; arXiv:2006.04424. https://arxiv.org/abs/2006.04424 — abstract only; stop logic from [75].
77. Eckhardt (KurtE), after Janssen and Halvorsen. `Arduino_Phoenix_Parts/Phoenix/Phoenix_Code.h` (master, accessed 2026-10-11). https://github.com/KurtE/Arduino_Phoenix_Parts/blob/master/Phoenix/Phoenix_Code.h — GaitSeq "return to home position"; Gait() lift at own slot if > 2 from home; NrLiftedPos + 1 extra cycles.
78. MIT Biomimetic Robotics Lab. `Cheetah-Software/user/MIT_Controller/Controllers/convexMPC/ConvexMPCLocomotion.cpp` (master). https://raw.githubusercontent.com/mit-biomimetics/Cheetah-Software/master/user/MIT_Controller/Controllers/convexMPC/ConvexMPCLocomotion.cpp — L31 standing gait all-stance; L119–129 stand_traj latch; p_rel_max 0.3.
79. Sapkal, Mancini, Kumar, Spiller, Murakami, Chiappe, Bidaye, et al. *Neural circuit mechanisms underlying context-specific halting in Drosophila.* Nature 634:191–200, 2024. https://doi.org/10.1038/s41586-024-07854-7 (opened at https://pmc.ncbi.nlm.nih.gov/articles/PMC11446846/) — walk-OFF relaxed stance; BRK finishes swing then freezes.
80. Daun-Gruhn, Büschges, Tóth. *A neuro-mechanical model containing fast and slow muscle fibres applied to mimic stop and re-start of a stepping leg.* BMC Neuroscience 14(Suppl 1):P86, 2013. https://doi.org/10.1186/1471-2202-14-S1-P86 — stance completed to static posture; restart by femur angle (abstract).
81. Schilling, Hoinville, Schmitz, Cruse. *Walknet, a bio-inspired controller for hexapod walking.* Biological Cybernetics 107(4):397–419, 2013. https://doi.org/10.1007/s00422-013-0563-5 — Rule 1 swing inhibition; standing without explicit desired positions; convergence from arbitrary configurations.
82. Schilling, Cruse. *neuroWalknet, a controller for hexapod walking allowing for context dependent behavior.* PLoS Computational Biology 2023. https://doi.org/10.1371/journal.pcbi.1010136 — Stand/Walk motivation units; Stand sets global velocity to zero.
83. Krabby project. *Close — p2r stand regime* and CHARTER §12. `crab_hex_omni_task/experiments/2026-10-07_omni_phase2/P2R_CLOSE.md` (lessons 1–12; Z1/Z2/Z2c/Z2d table; `p2r_rstg_read.json`, `p2r_z2c_read.json`, `p2r_z2d_read.json`, `p2r_z2d_2b.json`).
84. Krabby project. *Plan — a dedicated STAND network* and *gate calibration*. `crab_hex_omni_task/experiments/2026-10-11_stand_network/PLAN.md` (D1–D6, budget), `calib.md` (D6 table 15/15, R onset 0.12 s, R field_hard 0.825 n = 400, identity bridge PASS).

## Appendix: unverified or dropped items

- **Yu, Acero, Atanassov, Yang, Havoutis, Kanoulas, Li, *Discovery of Skill Switching Criteria for Learning Agile
  Quadruped Locomotion* (arXiv:2502.06676).** Exists and was opened; DROPPED as support for the hard-switch claim
  because its own method is a multiplicative blend π ∝ ∏π_i^{w_i(s,g)} of frozen skills with only the gate trained in
  the loop, reporting "smoother and continuous gait transitions" on hardware than a *manual* hard switch whose
  failures ("improper gait phase", bounding→galloping shutdown) the sweep quoted. Carried in §3 as the one blend
  result; it does not compare against a learned-selector hard switch.
- **Chen, Zhou, Koltun, Krähenbühl, *Learning by Cheating* (CoRL 2019, arXiv:1912.12294).** Exists; DROPPED from §4
  because the sweep misstated Table 1 as a cumulative chain: two-stage 16, +on-policy only 64, +white-box labels only
  (off-policy) 96, both 100; the DAgger stage alone reaches the same final accuracy, slower. Read correctly it says
  dense white-box labels on off-expert states recover most of the gap even offline, which weakens "offline cannot
  work" (R is white-box and can label counterfactual states).
- **Siekmann, Godse, Fern, Hurst, *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward
  Composition* (ICRA 2021).** Exists; DROPPED from §7 as an "imitation student with clock inputs": it is pure RL with
  periodic rewards and no references. Its clock inputs plus a gated standing cost (terms × ω switched off in the
  standing region) remain an RL precedent for mode-gated stand terms.
- **van Marum "most RL walkers have no native stand mode".** Corrected to "in many cases" (§2, [5]).
- **Lee et al. 2019 "switches seamlessly".** Corrected to "smoothly switches ... without any noticeable delay"; the
  paper does not state argmax vs sampling at deployment (§3, [2]).
- **Reske et al. BC 6.8 s vs MPC-Net 18.9 s survival as on-policy-vs-offline evidence.** Corrected: the BC baseline
  swaps the Hamiltonian loss for MSE, and the authors credit the Hamiltonian; not cited for this (§4, [11]).
- **Khadiv et al. "offline BC failed on dynamic gaits".** Corrected: offline BC learned jump and bound with PD
  actions; the failures were torque action spaces and end-to-end architectures (§4, [25]).
- **Reske "phase inputs cut constraint violation by a factor of four".** Corrected: φ, φ̇ were added to an existing
  sin(πφ) input (§7, [11]); the locator's CE(q̄, p) is the prior–posterior term, not the guided loss.
- **MoE-Loco cosine −0.132 vs 0.278 as the general gradient conflict.** Corrected: the worst cross-embodiment pair;
  other pairs differ little and one favours the MLP (§7, [57]).
- **Wen et al. 2021 "prove" copycat is loss-optimal.** Corrected: an argument the authors call "not fully rigorous"
  (§6, [43]). "Offline fixes rank below on-policy data" corrected: they beat DAgger on CARLA and lose on MuJoCo.
- **Pan et al. "rarely explored by the expert".** Paraphrase of "rarely explored in the expert's demonstrations" (§4,
  [18]).
- **Codevilla 2019 "a car never leaves 'brake'".** Loose: the inertia failure is a stopped car with no throttle; de
  Haan's brake-light example is a different confound (§6, [40, 38]).
- **Yang et al. DIR "rather than duplicating exact rows".** Inference: the paper compares against SMOTER/SMOGN, not
  exact duplication (§6, [46]).
- **IFM "12 % fall-offs for DAgger-only".** The figure covers MPC and DAgger together (§9, [24]). TWIST's ">>>" is
  ">>" in the fetched text; its λ schedule is unstated [65].
- **Isaac Lab "Spot joint_pos penalty only when stationary".** Corrected: always on, scaled ×5 when stationary (§2,
  [71]). **OpenSHC "every leg returns to default".** Corrected: a leg that fails its one attempt accepts its current
  tip (§10, [75]). **Cheetah "never re-places feet"** is inferred from the all-stance schedule (§10, [78]).
- **Khadiv perturbation "at episode starts".** Corrected: at every MPC re-plan point (§8, [25]).
- **Block et al. as executed-action noise.** Corrected: observation-input noise; the paper discourages action
  noising (§8, [63]). "Noise on the executed action, never on the label" is a synthesis rule, stated by no source.
- **Phoenix restart alignment ("try to align our Gaitstep to start with a good foot").** Inside `#ifdef QUADCODE`,
  the quadruped build, not the hexapod default; not cited (§10, [77]).
- **neuroWalknet stopping details.** In the S1 supplement, not opened; only the motivation-unit statement is cited
  [82]. **Daun-Gruhn et al.** is a conference abstract about a model, no new behavioural data [80]. **OpenSHC paper**
  abstract only; stop logic from the source file [76, 75].
- **Inferences, not source statements:** that Spencer's HARD regime applies to R + a smooth MLP; that R's 97 % probe
  does not measure recoverability from student states; that a missed LIFT onset is Swamy's single-state Ω(κT²) case;
  that the executed noised action in the history slot acts as de Haan's ORIGINAL intervention; that a phase-timed R_s
  is a candidate smoother; that masking deletes the phase information R's branch logic needs while the prop view
  retains it; the rung ordering and the "≤ 2 + 0.8 GPU-h" cost of rung 5.
- **In-repo numbers not re-measured here:** every campaign figure is quoted from P2R_CLOSE.md, PLAN.md and calib.md
  [83, 84] with their own verification tags (calib.md: 15/15 cells land, identity bridge PASS, 0 pending).
- **Unrefereed sources used with that caveat:** [3, 4, 10, 19, 23, 29, 42, 47, 57, 58, 66, 67] (arXiv preprints),
  [31, 68–71, 75, 77, 78] (source code), [49] (technical report), [14] (project page), [80] (abstract).
