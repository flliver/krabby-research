# Literature review: a traversable rough-terrain teacher for a heavy omnidirectional hexapod -- terrain envelope, omni-command curricula, scan use, loose ground and warm starts -- mapped to the Krabby campaigns

*2026-11-10. Prepared after two events. The construction-site teacher campaign closed not accepted (`crab_hex_omni_task/experiments/2026-10-28_site_teacher/REPORT.md`, decision 4, 2026-10-31). The rod-speed DR campaign then baked TDR as the omni teacher of record (`crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr/REPORT.md`). The plan for a traversable-terrain teacher is not written yet: slopes, hills, steps and ledges up and down, potholes, loose ground and rubble, with no walls and no stopping. This review builds on three earlier ones, which it does not repeat: `lit-review-construction-terrain-dr.md` (CT), `lit-review-obstacle-avoidance.md` (OA) and `lit-review-domain-randomization.md` (DR) [109–111]. The review's verification pass opened every citation and checked it against the paper's abstract or full text, or the raw source file, at time of writing. Where verification corrected a sweep claim, the corrected wording is used. Items that failed verification or are only inferred are in the appendix rather than softened. 12 / 12 clustered claims held or were corrected (64 sweep findings, 18 gap findings). An independent final check (Fable 5.2) read the written review against the verified evidence and raised 15 findings, none overturning a section; all 15 were applied (a promotion-cap formula, two absence claims restated as search results, column shares that summed past 100 %, the 1.05 m walking-height basis, and wording that had widened sources).*

## 1. The problems this review targets

Restated from the campaign ledgers [107, 108] so this document stands alone:

1. **The per-kind curriculum stalled.** Over 14 000 hazard iterations no primary held a mean level above ≈ 2.0 (a 0.134 m trench, a 0.098 m drop). The eval band sat at levels 5.4–9 (trench 0.31–0.45 m, drop 0.23–0.35 m). The measured stall mechanisms (REPORT, "Why gate b failed") [107]:
   - Promotion votes were a driftless random walk (trench up 0.333, down 0.350).
   - A crossing needed the root ≥ 2.96–3.2 m from a centre spawn on a ring. Under the 4-slot omni schedule that happened in only 0.40–0.45 of episodes.
   - Policy std 0.81 gave a training fail rate of 0.42–0.45 on rungs that the deterministic eval completed at 0.885 / 0.995.
   - Each kind had 1/20 of the columns (≈ 13 envs).
   - Every resume restarted levels at U{0..3}.
   - The void penalty came to ≈ −0.01 per episode, against ≈ 0.79 from tracking.
2. **Both teachers walked blind into hazards** at or above command speed. One loaded front foot went over the void, and pitch hit the 0.5 rad termination in a median 0.3 s (trench) or 0.6–0.8 s (ledge). Primaries were 0 / 200 for both teachers, rim stops 1 / 197 and 0 / 199, and the slowdown share ≤ 0.5 % [107]. On the holes row (0.36–0.72 m pits, 0.26–0.38 m deep) T\* completed 0.11 with 89 / 100 falls (CT §1) [109].
3. **The one real gain is unattributed.** On ledge_down d 0.3–0.6 (≈ 0.125–0.23 m drops) completion rose 0.29 → 0.59 (CI of the difference 0.23–0.37). At d 0–0.3, T\* already read 0.98 (ledge) and 0.785 (trench). Gate d's saliency tracked scan distribution shift, and no low-band flat-scan twin ran [107]. A 10 cm bump moves T\*'s actions by only 0.002–0.010 [109].
4. **Teacher and plant.**
   - TDR = T\* + 3000 iterations with rod-speed DR U(0.6, 1.35), 1.22 h wall-clock, std 0.80. In its no-regression block, field_hard read 0.96 against T\*'s 0.87 [108].
   - Layout v1 scans 9 × 12 rays at 0.3 m (1149-dim observation). Layout v2 (unmerged) scans 23 × 31 rays at 0.15 m plus a 5 × 5 patch at 0.08 m under each foot (1924-dim). v2 also re-slices v1 blocks and normalises mass (DESIGN §4.3).
   - Single-leg reach is 0.46 m down, 0.82 m up, and 0.88–1.05 m across a gap (`fk_reach.json`).
   - The body is ≈ 135–160 kg in sim (230 kg as built), with payload 0–180 kg (max_stable 135 kg), a nominal walking root height of 1.05 m (`fk_reach.json` h0; TDR's stand posture holds ≈ 1.12 m) and commands of 0.22–0.25 m/s. The robot overshoots back commands by 1.135× [107].

**The short answer.** Size the terrain to the robot. The 0.46 m drop reach and the friction bound on slopes bind before any precedent ratio does, and this search found no RL rough-terrain controller for a purely legged robot above 100 kg (§2). Centre-spawn pyramids put the feature in front of every heading (§3). Four changes each remove one stall mechanism: an integrated-command distance bar, tracking-based success for turn and stand slots, windowed or promote-only decisions, and levels that persist across resumes (§4). Slopes, hills, loose ground and low rubble are blind-solvable. Wide potholes and drops need a scan finer than 0.3 m (§5). Three levers for a lost foothold cost less than perception: slow the stance, keep four feet loaded, and loosen the termination (§6). Counterfactual scans and pre-contact foothold metrics show scan use (§7). Loose ground can be modelled as low-friction rough ground or a per-foot wrench (§8). A conservative TDR warm start keeps old skills, and v2 can be appended without changing the policy's outputs (§9). Evaluation should separate deterministic capability from curriculum statistics (§10).

## 2. Terrain envelope for a heavy, slow, statically stable hexapod

**Key papers:** Isaac Lab [1–3]; legged_gym [4, 5]; Lee et al. [6]; Miki et al. [7]; LP-ACRL [8]; UniPoint [9]; Spot [10, 11]; Unitree B2 [12]; hexapods [13, 15]; Kalakrishnan et al. [16]; Walas & Belter [17]; Komatsu et al. [18]; heavy machines and payload [19–39].

**Open recipes and ratings.**
- Isaac Lab's ROUGH_TERRAINS_CFG (8 m tiles) gives a 0.2 share each to pyramid and inverted stairs (0.05–0.23 m), boxes (0.05–0.20 m) and random rough (0.02–0.10 m), and 0.1 each to pyramid and inverted slopes 0–0.4 [1]. The docstring says radians, but the code uses rise over run, so 0.4 ≈ 22° [2]. Against ANYmal's 0.6 m spawn height, the steps reach 0.38 [3].
- legged_gym gives stairs 60 % of tiles and pits and gaps none [4, 5].
- Other recipes: Lee, hills of 0.2–3.0 m amplitude with roughness ≤ 0.05 m and steps of 0.05–0.3 m [6]; Miki, steps to 0.4 m [7]; LP-ACRL, stairs 0.05–0.20 m and slopes 0–0.40 [8]; UniPoint, platforms to 0.85 m [9].
- Ratings: Spot is rated for 300 mm steps and ±30° slopes (0.36 of standing height) [10, 11], B2 for 40 cm steps forward (≈ 0.62) [12]. Hexapods reached 0.49–0.67 of their height; a 25.5 kg one walked 3–20 cm stairs and 5–30° slopes blind [13, 15].
- In leg-length terms, LittleDog negotiated unperceived obstacles up to ≈ 30 % of its usable leg and needed planned footholds at 76 % [16]. A hexapod can climb obstacles as high as its stretched leg [17].

**Statics, payload and heavy machines.**
- Quasi-static equilibrium gives a necessary slope bound: tan θ ≤ μ for the friction cone, or μ/√2 along a pyramid axis, whatever the load split. This bound is our derivation from Komatsu's equations; their 100 kg simulation uses 10° at μ 0.8 and sets body height and tilt first, for leg reach [18].
- Slope capability depends on direction. A ≈ 4000 kg hexapod climbs ≈ 30° but traverses ≈ 41° along the contour [19]. HEAP (12 t, wheeled-legged) balanced on 100 % grades but left stepping to future work [20].
- Load narrows what the robot can hold. Torque limits shrink the usable support polygon [21], and payload height raises the CoM [22]. A payload-agnostic Go1 "eventually comes to a stop" on stairs under high payload [23]. Barry's 2:1 payload ratio is reported on flat terrain [24]. The first demonstration of 1 m/s locomotion on a hydraulic quadruped over 300 kg (an RL controller) was on flat ground [25].
- Heavy classical walkers state step limits at 0.35–0.42 of overall height, mostly as design specifications [26–32]:
  - ASV (≈ 2700 kg, 3.05 m tall) steps over a 1.22 m wall and is credited with crossing 2.7 m ditches.
  - COMET-IV (2200 kg) is specified for 1 m steps and 20° slopes, with a 0.28 m/s (1 km/h) maximum walking speed.
  - Dante II's obstacle clearance is 0.42 of its height.
  - Ambler was designed for 30° slopes "while crossing meter sized features".
- ElSpider (300 kg plus 150 kg load) publishes no terrain envelope [33]. TITAN XI (7 t) works steep slopes on wires [34].
- Learned control on heavy, purely legged robots is limited to RL gait modulation on flat ground [35], heuristic control in simulation [36], or no learning stated [37]. The RL exceptions are hybrids: a > 340 kg wheeled-legged robot on gravel, tested under slow actuation [38], and LEVA, whose payload tests used its rolling controller [39].

**What it means here.**
- The 3–9 cm field is 0.03–0.09 of Krabby's 1.05 m walking height.
- Precedent puts the top step-up near 0.37–0.44 m, under the 0.82 m rise reach.
- Drops and pothole depth hit the 0.46 m drop reach first. The site primaries (trenches at 0.67–0.98 of that reach) sat at this ceiling.
- Applying LittleDog's 0.3-of-leg blind limit to the drop reach gives ≈ 0.14 m. That is near where T\* fell from 0.98 to 0.29, at 0.125 m (inference).
- At μ 0.5 the slope bounds are 19.5° (pyramid) and 26.6° (cone); at the site DR's μ 0.25 floor they are 10.0° and 14.0°. Slope and friction must therefore be sampled jointly.
- This search found no RL precedent for carrying a payload near body mass over rough terrain.

## 3. Layout: meeting the feature from every heading

**Key papers:** Rudin et al. 2021 [40]; Isaac Lab [2, 41, 42]; legged_gym [4, 5]; Miki [7]; Yang, Gao & Li [14].

Rudin et al. build slopes and stairs as pyramids "to allow traversability in all directions" and spawn at the centre [40]. Isaac Lab's inverted variants put "the platform ... at the bottom" [2]. Its discrete obstacles, in the default "choice" mode, draw heights from {−h, −h/2, +h/2, +h}, so half are depressions [2]. Its "holes" variants confine features to a plus-shaped strip [41], and its pit terrain is one central pit [42]. legged_gym's pit is the sunk spawn square [4]. Heightfield edges are not vertical. legged_gym turns slopes above 0.75 into vertical faces [5], and Miki built stairs from boxes because the policy "exploited these non-vertical edges in simulation" [7]. A hexapod block map seeds robots on one side of the map, but its lateral commands were held at zero [14].

**What it means here.** Slope-up, slope-down, step-up (inverted pyramid) and step-down (pyramid) tiles with a centre spawn put the feature within one command hold in any heading, with no walls. This replaces the 2.96–3.2 m ring crossing. No stock generator scatters potholes on walkable ground, so Krabby's scatter-pit field (CT §5) stays, with finite floors and vertical rims built from boxes.

## 4. Promotion that votes under omnidirectional commands

**Key papers:** legged_gym [43, 5]; Isaac Lab [44, 45]; mjlab [46, 47]; Extreme Parkour code [48]; Robot Parkour Learning code [49]; Rudin et al. 2022 [50]; Lee [6]; Walk These Ways [51]; LP-ACRL [8]; PGTT [52]; DeXtreme [53]; Zhang et al. [54]; Kim et al. [55]; BeamDojo [56]; ALP-GMM [57]; ProCuRL [58].

**Distance rules.** The stock rule (legged_gym, Isaac Lab `terrain_levels_vel`) promotes when the root ends more than half a tile (4 m on 8 m tiles) from its env origin. It demotes when the distance is below 0.5 × |cmd_xy| × episode length [43–45]. The rule runs before the command reset, so a stand or yaw-only last command can never demote, and promotes only through 4 m of drift. Stock commands are symmetric U(−1, 1) m/s with 2 % standing envs and 20 s episodes [44, 5]. At 0.22–0.25 m/s, 20 s of straight walking covers 4.4–5 m, so the 4 m bar is 80–91 % of the commanded path. Command-scaled alternatives exist:
- mjlab integrates the commanded displacement over the episode and promotes above 0.8 of it, capped at half a tile. It never promotes an env commanded less than 1 m, "so drift cannot promote an env that was asked to stand still". It demotes below 0.5 ×, measures from the episode start, and ignores yaw [46, 47].
- Extreme Parkour uses 0.8 / 0.4 of the forward command × episode length [48].
- Robot Parkour Learning can require a time-out to promote, and a command above a cutoff to demote [49].
- A position-target formulation counts a terrain as solved when the robot ends within 0.5 m of a target in any direction [50].

**Tracking criteria cover yaw.**
- Lee's label (velocity along the commanded direction > 0.2 m/s) keeps terrains at traversability [0.5, 0.9], scored from the training rollouts; uniform sampling tested significantly worse. The label scores turn and stop commands as 0 [6].
- Walk These Ways passes a command bin when each tracking reward clears a fraction of its scale (lin 0.8, ang 0.5) [51].
- LP-ACRL (Isaac Lab) crosses v_x and yaw-rate levels with terrain and difficulty, 600 tasks in all. Success means more than 900 alive steps with EPTE-SP below 30 % on both v_x and ω_z; a fall charges every remaining step at worst-case error [8].
- PGTT passes a level when Gaussian tracking scores on v and ω are both ≥ 0.65 over dedicated evaluation agents, under U[−1, 1] commands (MJX) [52].

**Noise-tolerant rules never move a level on one episode.**
- DeXtreme averages up to 256 outcomes per boundary with a dead band, clears the queue when the bound moves, and gives 40 % of envs to evaluation [53].
- Zhang et al. demote only when the policy cannot beat random actions [54].
- Kim et al. ratchet each terrain's maximum stage upward on a traversability threshold and sample uniformly below it, with no demotion; they report this prevented catastrophic forgetting [55].
- BeamDojo advances a robot after three consecutive crossings [56].

Stock terrain shares are fixed per column [4, 45]; adaptive samplers follow learning progress instead [8, 57]. Estimating success from extra rollouts costs environment steps [58]. No locomotion source promotes from a deterministic eval subset.

**What it means here.** Each stall mechanism has a published fix:
- Measure from the episode start, with the bar set at 0.8 × integrated commanded displacement, capped at half a tile [46].
- Translate slots commanded less than 1 m abstain.
- Turn, move_turn and stand slots vote by survival plus per-axis tracking (EPTE-SP < 30 % [8]). Scoring stand by survival plus bounded drift is our inference.
- Decide per (kind, level) over a window [53], or use a promote-only ratchet [55], so std-driven failures (0.42–0.45 against 0.885 / 0.995) cannot demote.

`KRABBY_TERRAIN_PROMOTE 0.45:0.25` (CT §11) is already a tracking-ratio rule of the WTW / PGTT family and can be applied per command category.

## 5. Which terrains need the scan, and what scan

**Key papers:** UniPoint [9]; WMP [59]; Miki [7]; Lee [6]; RMA [60]; Robust Robot Walker [61]; Liu et al. [15]; LAURON V [62]; Weaver [63]; Isaac Lab Spot [64]; Jiang et al. [65]; He et al. [66]; Fan et al. [67]; NVM [68].

**Blind versus perceptive.** Matched comparisons agree that slopes and low discrete obstacles are nearly blind-solvable, and that gaps, drops and platforms are not:
- UniPoint, hardest level, blind vs full: slope 81.8 vs 83.5 %, discrete 93.6 vs 99.6 %, gap 21.3 vs 99.6 %, platform 12.1 vs 98.6 % [9].
- WMP returns: slope 33.96 blind vs 36.70, gap 9.80 vs 32.80 [59].
- Miki's perceptive policy crossed 30.5 cm steps; the blind baseline's success dropped at 20 cm [7].

Lee's blind policy cleared a 16.8 cm step, above its normal clearance, with a foot-trapping reflex, but "if the machine is commanded to walk off a cliff, it will" [6]. Loose ground has no matched comparison; it rests on blind demonstrations (RMA on sand, mud and memory foam) [60]. Robust Robot Walker learned 0.05–0.2 m pits blind, but with goal commands; its velocity-command variant scored 0.000 on the Mix benchmark [61]. Hexapods walked slopes and obstacles blind or reactively [15, 62], and Weaver adapted its gait from stereo roughness descriptors [63]. Isaac Lab runs 2–5 cm cobblestone with no height scanner [64].

**Scan footprints and noise.**
- Miki samples five rings per foot (radii 0.08–0.48 m), with clean heights for the teacher and noise regimes for the student [7].
- UniPoint scans 7 × 3 points per sole at ≈ 0.04 m; removing that scan cost stairs 99.5 → 92.4 % and stones 79.2 → 67.8 % [9].
- An omni stair humanoid uses a 1.4 × 1.0 m map at 0.05 m [65].
- He et al. give the example that at 10 cm resolution, stepping stones should be larger than 10 cm. They add noise and disturbances in a fine-tune stage while the critic stays privileged [66].
- An aggregated elevation map's success fell 51.7 → 24.3 % under odometry noise, degrading more than foot-centric sensing [67].
- Memoryless camera policies place front legs but not rear legs [68], and limited-memory controllers fail most "when the hind legs encounter the step" [6].

**What it means here.**
- Blind-solvable for Krabby: slopes, hills, gravel, rubble, and steps below ≈ 0.12–0.14 m (inference).
- Scan-separating: potholes wider than a foot, larger drops (the rows T\* fails), and step-ups above ≈ 0.12–0.14 m (platforms separate as strongly as gaps [9]; Miki's blind baseline dropped at 20 cm [7]).
- Flat-scan twins on the blind rows should show no gap.
- v1's 0.3 m pitch puts a 0.36 m pit on only 1–2 rays (CT §6). v2's 0.08 m foot patches are inside precedent; its 0.15 m body grid is coarser than every cited body grid (0.05–0.10 m) [7, 9, 65, 66]. But each patch spans only ± 0.16 m, so whether it reaches the next foothold in every commanded direction has to be checked against the measured step length.
- Under omni commands every leg is a hind leg for some command category.

## 6. The lost foothold: hexapod levers beside perception

**Key papers:** Walknet [69]; Bläsing & Cruse [70, 71]; Fukuhara et al. [72]; Gîlcă [73]; Yang & Kim [74]; Yang [75]; Ambler [32]; VB-Com [76]; Robust Robot Walker [61]; Siekmann et al. [77]; terminations [43, 48, 78–80].

**How insects handle a missing foothold.** A leg that finds no ground searches, and the other legs slow down; the controller does not reach further [69–71]. Walknet turns a swing that ends without contact into a quasi-rhythmic search [69]. Stick insects cross gaps wider than twice their step length, and the authors name slowed stance, searching movements and short steps as prerequisites. A front-leg step into the gap is what triggers the slowing [70, 71]. In a 0.92 kg simulated hexapod, a load-sensing search raised gap crossing from < 30 % to ≈ 60 %. The remaining failures came from stance legs that kept retracting at constant speed until the CoM left the support polygon [72].

**Support rules.** Static stability needs at least 3 feet down. A tripod gait has exactly three, while wave gaits keep 4 or 5 (λ = 2/3, 5/6) [73]. Yang & Kim define a fault stability margin, the margin left after any one leg fails suddenly. They derive periodic gaits with four legs in support as optimal under it [74]; a fault-tolerant tripod has less margin than normal gaits [75]. The Ambler's gaits stay stable if any supporting leg fails [32].

**Learned recovery from unseen holes.** Where it was learned, holes came in a graded curriculum under blind or corrupted perception:
- VB-Com trains its blind policy with noise and delays up to 0.5 s. Its composite policy completes 84.8 % of goals at 100 % perception noise, against 48.7 % for the vision-only policy [76].
- Robust Robot Walker trained pits up to 0.30 m on a circle, with 8 s episodes, a dense goal reward and a contact encoder. Pit success was 0.902, against 0.796 for proprioception only and 0.000 for RMA, MoB and HIMLoco [61].
- Blind Cassie learned drop reflexes from randomised 10–21 cm stairs [77].

**Terminations.** legged_gym and Isaac Lab reset only on base contact and time-out [43, 78]. Extreme Parkour terminates at |roll| or |pitch| > 1.5 rad [48]. Yang et al. terminate at 90°, and trained fall recovery on the time limit alone "because the robot had to undergo the states" [79]. Binary early termination "entirely fails" in CaT's constraint problem [80]. No source ablates a tight pitch limit for hole recovery.

**What it means here.** Problem 2 is Fukuhara's failure on a 135 kg body: the stance set keeps propelling (rim stops 1 / 197) while a front foot has no support. Three levers cost little at 0.25 m/s:
- reward or permit a stance-speed drop after a front-foot touchdown miss;
- prefer ≥ 4 loaded feet on pothole and drop rungs;
- on those rungs, replace the 0.5 rad pitch limit, a third of Extreme Parkour's. It ends episodes 0.3–0.8 s after the void contact, which deletes the states in which reloading other legs could be learned (inference).

Robust Robot Walker's 0.000 baselines are a warning that proprioceptive history alone may not discover pothole recovery.

## 7. Measuring scan use and anticipation

**Key papers:** Miki [7]; VB-Com [76]; Extreme Parkour [81]; UniPoint [9]; Jiang et al. [65]; TAMOLS [82]; Walas & Belter [17]; Kalakrishnan [16].

Miki tested scan use with three counterfactuals [7]:
- a foam block that the map shows as solid: the robot prepares to step, then revises its belief on contact;
- a transparent obstacle the map misses: the robot walks as if on flat ground until contact;
- covered sensors.

VB-Com's perception-noise test (84.8 vs 48.7 % of goals at 100 % noise) [76] suggests a dose-response read; its graded series is not cited because that source failed verification (appendix). Pre-contact foothold quantities serve both as metrics and as dense rewards that only a scan can earn:
- Removing Extreme Parkour's 5 cm edge term raised mean edge violation 0.03 → 0.08 at unchanged displacement [81].
- UniPoint penalises stance-sole points outside a [−0.11, 0.02] m band [9].
- Jiang defines a hazard as a riser collision or more than 50 % of the sole over an edge, and reports a safe-stepping rate. With the penalty the curriculum reached its top level by ≈ iteration 3000; without it, it plateaued near level 6 [65].
- TAMOLS penalises |∇h|² at footholds on raw and blurred maps; a wider blur keeps feet farther from edges [82].
- Classical foothold features add signed concavity and slip angle [17] and drop-off distance [16].

**What it means here.** Replace gate d with four measures:
- flat-scan twins on scan-separating rows away from the floor and ceiling of the ladder (ledge_down d 0.3–0.6 was the missing one);
- phantom rows (a pothole in the scan over flat physics) and invisible rows (a real pothole under a flat scan), both of which keep proprioception in distribution [7];
- a corruption dose-response curve (inference from [76]);
- per row, the fraction of loaded feet over a void, the touchdown distance to the rim, and the approach-speed drop before the first front foot reaches the hazard (slowdown share is ≤ 0.5 % today).

An edge-band or blurred-gradient term computed from the generator's masks would give the dense gradient. The site campaign's 5 cm edge term, however, failed its replay and was never armed [107].

## 8. Loose and granular ground

**Key papers:** Lee [6]; GM-Loco [83]; Farook Deen [84]; Walas & Belter [17]; Kottege et al. [85]; Ambler [32]; Irawan & Nonami [86]; CT §8 [109].

CT §8 covers soft-ground models; this section adds only the verified essentials.
- Lee trained on rigid terrain only. That included Slippery Hills with friction N(0.3, 0.1) (N(0.7, 0.2) elsewhere), clipped above 0.1, roughness up to 0.05 m, and privileged per-foot friction. The controller handled mud, snow and rubble, but no ablation isolates which training terrain caused this [6].
- Particle gravel scales poorly. An Isaac Sim thesis reports that the PBD pipeline runs "entirely on the CPU", which limits frame rates, and that GPU memory and throughput limits stopped it scaling [84].
- GM-Loco applies a per-foot resistive-force wrench computed in Warp and trained 4096 Isaac Lab agents on one RTX 4090. In MPM evaluation a rigid-trained PPO humanoid scored 98 % on basalt and 0 % on sand, while the RFT-trained policy scored 100 % on all three media [83].
- Classical metrics and designs: per-foot slip is r_i = d_i / |f_i| [17], and proprioceptive power alone guided hexapod gait switching for traction [85]. The Ambler assumes "any leg might spontaneously fail to provide support" [32]. COMET-IV was validated on random hard bricks and rubber pads [86].

**What it means here.** The gravel row is low friction (N(0.3, 0.1), clipped at 0.1) plus 0.02–0.05 m roughness, with privileged per-foot friction. It is blind, cheap, and not a lever for scan use. A per-foot support-failure or sinkage event stands in for subgrade failure without particles, and also trains the response described in §6 (inference). Add a Warp RFT hook only if deep loose fill matters [83].

## 9. Warm start, a changed scan layout, and forgetting

**Key papers:** Rudin et al. 2025 [87]; He et al. [66]; rsl_rl [88]; Isaac Lab [45]; OpenAI Five [89]; Net2Net [90]; progressive networks [91]; Kickstarting [92]; Robot Parkour Learning [93]; SoloParkour [94]; Distillation-PPO [95]; PIRLNav [96]; Wolczyk et al. [97]; PPF [98]; MPC fine-tune [99]; Andrychowicz et al. [100].

**Fine-tuning without forgetting.**
- Naive RL fine-tuning "steadily degrades" a capable policy. Rudin et al. lowered the initial std, froze the policy while the critic pre-trained, and kept the old terrains in the mix. Every trained terrain then held or improved (stepping stones 73.0 → 98.8 %). A new terrain at only 3 % of samples rose 54.4 → 92.4 %, while old terrains moved by ≤ 0.6 points [87].
- He et al.'s 3600-epoch all-terrain fine-tune added noise and drift and lowered the entropy coefficient 0.005 → 0.002. Training on all terrains from scratch was the worst variant [66].
- Curriculum state is lost on resume: rsl_rl saves only algorithm state, the iteration counter and infos [88], and Isaac Lab redraws terrain levels at every build [45].
- In PIRLNav, a critic-only phase plus LR scheduling raised the fine-tune gain from +1.6 to +9.9 % [96].
- Fine-tuning forgets states that are not visited early. A KL to the frozen policy on stored pre-training states worked best, while kickstarting on current-policy states "fails completely" under a coverage gap [97]. A state-dependent anchor beat both a fixed anchor and none (14° tracking error 18.9 vs 28.7 vs 30.7) [98].
- Std values: a warm-started quadruped used std 1.0 (against 3.0 from scratch), LR 1e-5 and clip 0.05–0.1 [99]. From scratch, initial std 0.5 was best on all but one task [100].

**Adding inputs.** OpenAI Five ran more than 20 surgeries. It zero-initialised the new weights (Ŵ = [W 0]), held deprecated inputs constant, and set the LR to 0 after each one. A from-scratch rerun at ≈ 20 % of the compute beat the surgered agent more than 98 % of the time [89]. Net2Net growth "immediately performs as well as the original network" [90], and progressive columns are "immune to forgetting" [91].

**Re-distillation.** Kickstarting matched from-scratch performance in ≈ 10× fewer steps [92]. A DAgger parkour student kept 86/80/100/73/100 % success against its specialists' 95/82/100/100/100 % [93]. SoloParkour's warm-started RL beat DAgger and BC at equal compute [94]. A mixed distillation-plus-PPO loss reports only qualitative gains [95].

**What it means here.** TDR was itself a 1.22 h, 3000-iteration fine-tune that raised field_hard [108]. The from-scratch v2 chain cost 23.75 GPU-h and gained only on low rungs [107]. For the next fine-tune:
- Lower the initial std, as Rudin et al. did [87]. The one published legged warm-start value is 1.0 (3.0 from scratch) [99], above TDR's 0.80, and no source ablates it, so the value (e.g. 0.5) is to calibrate.
- Freeze the actor while the critic learns the new terrain mix.
- Anchor the actor to frozen TDR on stored flat and field states, with zero weight on new-terrain envs.
- Keep field, flat, payload and turn rows in the mix.
- Save per-kind levels in the checkpoint infos.

A v2 surgery preserves the policy's function only if the 1149 v1 inputs keep their exact v1 meaning. v2 re-slices blocks, normalises mass and moves velocity into the estimator slot [107]. So append the v2 scan and patches with zero-initialised weights, and check action equality at iteration 0. Zero-init does not create scan use by itself (§5–§7).

## 10. Evaluation

**Key papers:** PIE [101]; Rudin 2025 [87]; He et al. [66]; PGTT [52]; LP-ACRL [8]; Hirose et al. [102, 103]; Papadopoulos & Rey [104]; Zhang et al. [105]; Černý et al. [106].

- PIE advances robots from the easiest of ten levels until a fall or collision, with 40 sets × 100 robots per terrain [101].
- Rudin 2025 runs 1000 roll-outs at 90 % of the top training difficulty [87].
- He et al. classify outcomes as success, failure or stuck [66].
- PGTT reports the median and interquartile range over 5 seeds [52].
- EPTE-SP charges falls as worst-case tracking error [8].
- The normalised energy stability margin (the CoG rise needed to tumble, independent of weight) was the most accurate margin in Hirose's slope experiments [102, 103].
- The force-angle measure is sensitive to CoM height under heavy loads [104], and a heavy-duty hexapod uses it to detect tipping [105].
- On a small hexapod, only adaptive gaits crossed irregular blocks [106].

**What it means here.** Evaluate each terrain kind × command category with deterministic actions:
- level reached;
- success / fall / stuck at 90 % of the top rung ("stuck" matters when the robot must never stop);
- EPTE-SP per axis, which also catches the 1.135× back overshoot;
- each of these at 0 and 135 kg and at rod speed ×0.6–0.7;
- diagnostics: minimum NESM computed with the payload-inclusive CoM, and the foothold metrics from §7.

This separates the 0.42–0.45 training fail rate from the 0.885 / 0.995 capability, which the stall monitor confused.

## 11. Literature gaps

- This search found no RL rough-terrain controller for a purely legged robot above 100 kg [25, 33, 35–39], and no source showing step traversal under a payload near the robot's own mass [23, 24, 39].
- Omnidirectional terrain curricula at Isaac Lab scale rest on LP-ACRL alone [8]. No source scores stand slots or promotes from a deterministic eval subset [58].
- No source ablates tight orientation terminations for hole recovery [48, 79, 80]. This review did not search for RL hexapod pothole or gap results; its hexapod hole evidence is insect studies, reflex controllers and gait theory [69–75].
- There is no matched blind-vs-perceptive comparison on loose ground [60], and no ablation of resetting versus inheriting std in warm starts [99].
- Load-bearing preprints without replication: [8, 9, 15, 23, 52, 59, 61, 65–67, 76, 83, 87, 98, 99].

## 12. Ranked recommendations

Knobs: `KRABBY_OMNI_OBS_LAYOUT` (v1 | v2), `KRABBY_OMNI_SCAN_ABLATE` (off | flat | zero | freeze), `KRABBY_MAX_INIT_TERRAIN_LEVEL`, `KRABBY_TERRAIN_PROMOTE`, `KRABBY_SITE_PROMOTE_RULE` (the hook a new rule plugs into), rod-speed DR `0.6:1.35` [107, 108]. Cost basis (wall-clock, local GPU) [107, 108]:
- TDR's 3000-iteration fine-tune: 1.22 h;
- one 100-episode eval row: ≈ 0.06 h (site step 9 ran 78 rows in 4.8 h);
- the from-scratch v2 chain: 23.75 h;
- a student relaunch: 9.6 h.

0. **Sort the terrain set into blind and scan rows on TDR before training, ≈ 2–3 h; v1 as is** (§5, §7; [9, 59, 7]). Run TDR with `SCAN_ABLATE` off and flat on slope, hills, gravel, rubble, step-up, step-down and pothole rows, at three difficulties each. Expect no gap on the first four. Pre-register the scan statistic only on rows where blind TDR fails. Campaign: the +0.30 ledge gain never had a flat twin [107].
1. **An omni promotion rule, ≈ 0 h (replay it on CPU against the site-stage telemetry first); layout-independent** (§4; [46, 8, 53, 55, 54]).
   - Promote when distance from the episode start ≥ min(0.8 × integrated commanded displacement, half tile).
   - Translate slots commanded less than 1 m abstain.
   - Turn, move_turn and stand slots vote by survival to time-out plus EPTE-SP < 30 % on the active axes. For stand, survival plus bounded drift is our inference.
   - Demote below 0.5 × the commanded distance, as mjlab does [46]. Demoting only on hazard-caused falls is the site report's own lesson, not a published rule (inference).
   - Decide per (kind, level) over a window of outcomes, cleared when the level moves. DeXtreme used up to 256 outcomes with a dead band on consecutive successes [53]; a 64–256-outcome window with a 0.3–0.8 success-rate dead band is our inference. Or use a promote-only ratchet with uniform sampling below the frontier [55].
   - Persist levels in the checkpoint infos.

   Campaign: votes 0.333 vs 0.350, P(cross) 0.40–0.45, training fail 0.42–0.45 vs deterministic 0.885 / 0.995, U{0..3} resets [107].
2. **Pyramid layout, fewer kinds and a retention share; layout-independent** (§3; [40, 2, 87]). Use centre-spawn slope and step pyramids (normal and inverted), a scatter-pothole field with finite floors and vertical rims, hills and gravel. That is seven or eight kinds (four pyramids, potholes, hills, gravel, and the rubble of item 3) sharing ≈ 70–80 % of the columns, ≈ 1/11–1/7 each instead of 1/20, plus ≈ 20–30 % field and flat terrain (inference; a new terrain needed only 3 % of samples in [87]). Campaign: 2.96–3.2 m ring crossings, ≈ 13 envs per kind [107].
3. **An envelope normalised to Krabby** (§2; inference from [1, 6, 16, 18, 26, 29]). Top rungs:
   - slopes 20° (rise / run 0.36), with tan θ ≤ 0.7 μ enforced per env;
   - hills under the slope cap, with ≤ 0.05 m roughness;
   - steps up 0.40 m (0.38 of the 1.05 m walking height; rise reach 0.82 m);
   - steps down and pothole depth 0.35 m (0.75 of the 0.46 m drop reach);
   - pothole width up to ≈ 0.6 m (0.7 of the 0.88 m gap reach), on its own ladder;
   - gravel: μ N(0.3, 0.1) clipped at 0.1, with 0.02–0.05 m roughness;
   - rubble ±0.15 m.

   All ladders start at ≈ 0.03–0.05 m. Cap payload at 135 kg on the top two rungs. Drops and potholes above ≈ 0.14 m will likely need item 5. Campaign: 0.23–0.35 m drops read 0 / 200 for both teachers, and the only gain sat at 0.125–0.23 m [107].
4. **Warm-start TDR conservatively, ≈ 1.2 h per 3000 iterations; v1** (§9; [87, 96, 97, 98, 99]).
   - Start from an initial std below 0.80.
   - Run a critic-only phase until the value loss flattens.
   - Anchor the actor with a KL to frozen TDR on stored flat and field states, with zero weight on new-terrain envs.
   - Keep the rod-speed DR.
   - Retention gates: field_hard (TDR 0.96), field_light, suite tracking, side22, payload 135 kg.

   This disagrees with the site campaign's from-scratch route [107].
5. **v1 first; v2 by surgery only where item 0 finds scan-limited rows** (§5, §9; [89, 90, 92, 94]). Append the v2 blocks with zero-initialised weights, keep the meaning of every v1 input, and check action equality at iteration 0. Fallback: kickstart TDR into a v2 network, then run PPO. GPU cost as in item 4.
6. **Hexapod lost-foothold levers on pothole and drop rungs; either layout** (§6; [70–74, 79, 80]).
   - Add a ≥ 4-loaded-feet or fault-stability-margin term.
   - Permit or reward a stance-speed drop after a front-foot touchdown miss.
   - Terminate on chassis contact plus a looser pitch limit (≈ 1.0 rad, inference), with a graded tilt penalty.
   - Report the recovery rate after the first foot-over-void event.

   Campaign: pitch reached 0.5 rad in 0.3–0.8 s; rim stops 1 / 197 [107]. Replay on the saved fall npz first.
7. **Foot-placement income from the generator's masks; needs a scan that resolves rims (v2)** (§7; [81, 9, 65, 82]). Add an edge-band or blurred-gradient penalty at stance feet, ramped in after the critic phase. Replay it first, since the site campaign's 5 cm term failed its replay [107].
8. **Loose ground without particles; either layout** (§8; [6, 83, 32, 86]). Use low-μ rough rows, privileged per-foot friction, per-foot support-failure events, and a slip ratio logged per terrain and friction bin. Add an RFT hook only if deep loose fill is in scope.
9. **Evaluation as in §10; either layout.** Deterministic, per kind × category, at 0 / 135 kg and rod speed ×0.6–0.7, with counterfactual scan rows on separating kinds only. ≈ 0.06 h per row.

**Where the literature disagrees with the campaign's framing.**
- (i) A traversable-only terrain set is mostly blind-solvable, so it will not produce a scan teacher. Potholes, drops and step-ups above blind swing height (≈ 0.12–0.14 m, so item 3's upper step rungs) create that pressure [9, 59, 6, 7]. With stopping removed (site lesson 7), foot placement is the lever left.
- (ii) The literature supports noise-tolerant demotion: mjlab demotes below 0.5 × the commanded distance [46], and Zhang et al. demote only when the policy cannot beat random actions [54]. The site report's "demote only on hazard-caused falls" is stricter than either. The literature does agree with putting the hazard within one command hold [40].
- (iii) CT called for a pitch or impact termination on hard pits. That runs against an inference from fall-recovery training and early-termination results that tight terminations delete recovery states [79, 80]; no source tests a pitch limit on hole recovery. A slow, statically stable hexapod is not a quadruped jumping down.
- (iv) This search found no RL precedent for carrying a payload near body mass on such rungs [23, 24, 39].
- (v) A from-scratch rerun can reach a higher ceiling than surgery [89]. But He et al.'s from-scratch arm was their worst [66], and the site campaign's chain never reached its band. Wall-clock cost favours the warm start.

## References

1. Isaac Lab `terrains/config/rough.py` (main). https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/terrains/config/rough.py — ROUGH_TERRAINS_CFG: 8 m, 10 × 20, shares and ranges.
2. Isaac Lab `terrains/height_field/hf_terrains.py` and `hf_terrains_cfg.py` (main). https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/terrains/height_field/hf_terrains.py — slope × size/2 (rise over run) vs the "in radians" docstring; inverted "platform is at the bottom"; choice mode {−h, −h/2, h/2, h}; holes_depth −10.
3. Isaac Lab `isaaclab_assets/robots/anymal.py` (b0542fe2). https://raw.githubusercontent.com/isaac-sim/IsaacLab/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_assets/isaaclab_assets/robots/anymal.py — init pos z = 0.6.
4. leggedrobotics/legged_gym `utils/terrain.py` (master). https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/utils/terrain.py — slope 0.4 d; step 0.05 + 0.18 d; gap and pit 1.0 d; column choice by proportions.
5. legged_gym `legged_robot_config.py` (master). https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot_config.py — proportions [0.1, 0.1, 0.35, 0.25, 0.2]; slope_treshold 0.75; max_init_terrain_level 5; commands ±1; 20 s.
6. Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning Quadrupedal Locomotion over Challenging Terrain.* Science Robotics 5(47), 2020; arXiv:2010.11251. https://arxiv.org/abs/2010.11251 — Table S2 ranges and friction; Eqs. 2–7 traversability [0.5, 0.9]; 16.8 cm step; cliff quote; hind-leg failures; rigid-only training.
7. Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning robust perceptive locomotion for quadrupedal robots in the wild.* Science Robotics 7(62), 2022; doi:10.1126/scirobotics.abk2822. https://arxiv.org/abs/2201.08117 — S4 ranges; boxes for stairs; S5 rings; noiseless teacher; 20 vs 30.5 cm; foam, transparent and covered-sensor tests.
8. Li, Li, Hutter. *Scaling Rough Terrain Locomotion with Automatic Curriculum RL (LP-ACRL).* arXiv:2601.17428, 2026. https://arxiv.org/html/2601.17428v1 — 600 tasks; > 900 steps and EPTE-SP < 30 %; Eq. 8; App. C ranges; signed LP sampling.
9. Li, Chu, Li, Zhu, Wu. *UniPoint: Unified Point-Level Sensor Fusion for Humanoid Locomotion Across Challenging Terrains.* arXiv:2609.23666, 2026. https://arxiv.org/html/2609.23666 — Table V blind vs full; 7 × 3 sole scan; band [−0.11, 0.02] m; terrain ranges.
10. Boston Dynamics. *Spot specifications sheet.* https://bostondynamics.com/wp-content/uploads/2020/05/spot-spec-sheet.pdf — 840 mm standing; 300 mm step; ±30°.
11. Boston Dynamics. *Spot product page.* https://bostondynamics.com/products/spot/ — 610 mm default walking height.
12. Unitree. *B2 product page.* https://www.unitree.com/b2 — ≈ 645 mm; 40 cm steps forward; > 45°.
13. Qu, Li, Zakhor, Yu, Zhang. *Versatile Locomotion Skills for Hexapod Robots.* arXiv:2412.10628, 2024. https://arxiv.org/html/2412.10628v1 — risers 4.5–18 cm; ≈ 37 cm tall.
14. Yang, Gao, Li. *Terrain-adaptive Central Pattern Generators with RL for Hexapod Locomotion.* arXiv:2310.07744, 2023. https://arxiv.org/html/2310.07744v1 — 20 × 10 block map; one-side start; lateral command 0.
15. Liu, Wu, Li, Qi, Xue, Gao. *Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors based on DRL.* arXiv:2511.03167, 2025. https://arxiv.org/html/2511.03167v1 — 25.5 kg, 30 cm; proprioceptive actor; 3–20 cm stairs, 5–30°.
16. Kalakrishnan, Buchli, Pastor, Mistry, Schaal. *Fast, Robust Quadruped Locomotion over Challenging Terrain.* ICRA 2010. https://fileadmin.cs.lth.se/ai/Proceedings/ICRA2010/MainConference/data/papers/1663.pdf — 4 cm unperceived on a ≈ 13 cm leg; 76 %; drop-off templates.
17. Walas, Belter. *Supporting locomotive functions of a six-legged walking robot.* Int. J. Appl. Math. Comput. Sci. 21(2):363–377, 2011; doi:10.2478/v10006-011-0027-9. https://zbc.uz.zgora.pl/Content/46922/download — stretched-leg obstacles; K1–K3; slip r_i = d_i/|f_i|.
18. Komatsu, Endo, Hodoshima, Hirose, Fukushima. *Basic Consideration about Optimal Control of a Quadruped Walking Robot during Slope Walking Motion.* IEEE ARSO 2013; doi:10.1109/ARSO.2013.6705533. https://t2r2.star.titech.ac.jp/rrws/file/CTT100663998/ATD100000413/ — quasi-static equilibrium; Eqs. 9–10 friction pyramid; 10°, μ 0.8, 100 kg.
19. Tsumaki, Tamada. *Development of the Stable Walking System for the Large 6 Legged Walking Machine.* JRSJ 35(6):486–493, 2017; doi:10.7210/jrsj.35.486. https://www.jstage.jst.go.jp/article/jrsj/35/6/35_35_486/_article/-char/en — ≈ 4000 kg; ≈ 30° climb, ≈ 41° lateral.
20. Jud, Kerscher, Wermelinger, Jelavic, Egli, Leemann, Hottiger, Hutter. *HEAP — The autonomous walking excavator.* Automation in Construction 2021; arXiv:2106.05059. https://ar5iv.labs.arxiv.org/html/2106.05059 — 12 t; quasi-static; 100 % gradients; stepping future work.
21. Orsolino, Focchi, Caron, Raiola, Barasuol, Semini. *Feasible Region: an Actuation-Aware Extension of the Support Region.* IEEE T-RO 2020; arXiv:1903.07999. https://arxiv.org/abs/1903.07999 — torque-infeasible CoM positions; HyQ + 10 kg.
22. Boston Dynamics. *Payload Configuration Requirements.* https://dev.bostondynamics.com/docs/payload/payload_configuration_requirements — payload height raises the CoM, hurts self-righting.
23. Kurva, Kolathaya. *MULE: Multi-terrain and Unknown Load Adaptation for Effective Quadrupedal Locomotion.* arXiv:2505.00488, 2025 (preprint). https://arxiv.org/html/2505.00488v1 — baseline "eventually comes to a stop" on stairs.
24. Valsecchi et al. *Barry: A High-Payload and Agile Quadruped Robot.* IEEE RA-L 8(11), 2023; doi:10.1109/LRA.2023.3313923; coverage https://spectrum.ieee.org/quadruped-robot-dog — 48 kg; 90 kg unmodeled payloads; 2:1 on flat terrain.
25. Lee, Kim, Kim, Park, Lee, Cho, Hwangbo. *Learning Quadrupedal Locomotion for a Heavy Hydraulic Robot Using an Actuator Model.* IEEE RA-L 10(12), 2025; arXiv:2601.11143. https://arxiv.org/abs/2601.11143 — > 300 kg; flat training and tests.
26. Ohio State University Libraries. *Adaptive Suspension Vehicle.* https://library.osu.edu/site/archives/?p=2453 — 9-ft ditches, 7-ft walls.
27. Shvachko. *Adaptive Suspension Vehicle* (citing Waldron & McGhee 1986). https://shvachko.net/?p=1088 — 2700 kg; leg stroke 1.80 m.
28. Song, Waldron. *Machines That Walk: The Adaptive Suspension Vehicle.* MIT Press (publisher description). https://www.schweitzer-online.de/en/buch/Song/Machines-Walk/9780262192743/A1380658/ — 10 ft high; steps over a 4 ft wall.
29. Ohroku. *COMET-IV teleoperation* (slides, Chiba University). https://www.slideshare.net/slideshow/outline-3980941/3980941 — 2200 kg; 2.8 m; step 1 m; 20°; 1 km/h.
30. cyberneticzoo. *Dante & Dante II.* https://cyberneticzoo.com/walking-machines/1992-4-dante-dante-ii-john-e-bares-william-red-whittaker-american/ — 120 in high; 50 in clearance.
31. Simmons, Krotkov, Whittaker et al. *Progress towards robotic exploration of extreme terrain.* Applied Intelligence 1992. https://publications.ri.cmu.edu/progress-towards-robotic-exploration-of-extreme-terrain — 30° while crossing meter-sized features.
32. Mahalingam, Dwivedi. *Periodic gaits for the CMU Ambler.* NASA NTRS 19900019711. https://ntrs.nasa.gov/api/citations/19900019711/downloads/19900019711.pdf — any supporting leg may fail; ≈ 1 m/min.
33. Ding, Xu, Gao, Wang, Zhou, Gong, Liu. *Fault Tolerant Free Gait and Footstep Planning for Hexapod Robot Based on Monte-Carlo Tree.* arXiv:2006.07550, 2020. https://arxiv.org/abs/2006.07550 — ElSpider 300 kg, 150 kg load.
34. Saitama University. *TITAN XI.* https://design.mech.saitama-u.ac.jp/worldwide/research/titanxi/titanxi.html — 7 t; two force-controlled wires on slopes.
35. Wu, Gao, Xie. *Locomotion Design and Experimental Validation of a 10-Ton-Class Firefighting Hexapod Robot.* ICCAR 2026; doi:10.1109/ICCAR69571.2026.11549880. https://pure.bit.edu.cn/en/publications/locomotion-design-and-experimental-validation-of-a-10-ton-class-f-2/ — RL modulates gait parameters; flat ground.
36. Hou, Chai, Li, Xin, Chen. *A heuristic control framework for heavy-duty hexapod robot over complex terrain.* IET Cyber-Systems and Robotics 4(4):322–330, 2022. https://doi.org/10.1049/csy2.12064 — simulation only (abstract).
37. Moog Inc. *Moog and IIT Test New Quadruped Robot HyQReal by Pulling Three-Ton Airplane* (2019). https://www.moog.com/news/operating-group-news/2019/Moog-IIT-Istituto-Italiano-di-Tecnologia-Test-New-Quadruped-Robot-yQReal-Pulling-Airplane.html — 130 kg, 0.9 m; with ECHORD++ https://echord.eu/hyqreal.html — "careful walking over rough terrain", no learning stated.
38. Hou, Wang, Chai, Xu, Li, Song, Liu, Zhang. *Learning-Based Hybrid Locomotion Control for a Heavy Wheeled-Legged Robot in Challenging Terrains.* J. Intell. Robot. Syst. 2026. https://doi.org/10.1007/s10846-026-02423-8 — > 340 kg; gravel, wetlands; slow-response actuation (abstract).
39. Arnold et al. *LEVA: A high-mobility logistic vehicle with legged suspension.* ICRA 2025; arXiv:2503.10028. https://arxiv.org/html/2503.10028v3 — 85 kg; payload ratio one; payload tests with rolling controller.
40. Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL 2021, PMLR 164; arXiv:2109.11978. https://arxiv.org/html/2109.11978v3 — pyramids "to allow traversability in all directions"; centre spawn.
41. Isaac Lab `trimesh/mesh_terrains_cfg.py` (b0542fe2). https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains_cfg.py — holes: plus-shaped strip.
42. Isaac Lab `trimesh/mesh_terrains.py` (b0542fe2). https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py — one pit "with levels (stairs) leading out".
43. legged_gym `legged_robot.py` (master). https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot.py — `_update_terrain_curriculum`; contact > 1 N termination.
44. Isaac Lab velocity `curriculums.py`, `velocity_env_cfg.py`, `manager_based_rl_env.py` (v2.1.0; same rule at v2.3.2 and b0542fe2). https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py — move_up / move_down; defaults; curriculum before command reset.
45. Isaac Lab `terrain_importer.py` (main). https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/terrain_importer.py — contiguous column blocks; levels randint at build; top level to random.
46. mujocolab/mjlab `tasks/velocity/mdp/curriculums.py` (main). https://raw.githubusercontent.com/mujocolab/mjlab/main/src/mjlab/tasks/velocity/mdp/curriculums.py — promotion_fraction 0.8; min_commanded_distance 1.0; episode-start distance.
47. mjlab documentation. *Curriculum.* https://mujocolab.github.io/mjlab/main/source/curriculum.html — integrated command; half-tile cap; demotion below half.
48. chengxuxin/extreme-parkour `legged_robot.py` (main). https://raw.githubusercontent.com/chengxuxin/extreme-parkour/main/legged_gym/legged_gym/envs/base/legged_robot.py — 0.8 / 0.4 × cmd_x × episode length; roll / pitch > 1.5 rad.
49. ZiwenZhuang/parkour `legged_robot_field.py` (main). https://raw.githubusercontent.com/ZiwenZhuang/parkour/main/legged_gym/legged_gym/envs/base/legged_robot_field.py — 1.5 block lengths; time-out gate; lin_cmd_cutoff.
50. Rudin, Hoeller, Bjelonic, Hutter. *Advanced Skills by Learning Locomotion and Local Navigation End-to-End.* IROS 2022; arXiv:2209.12827. https://arxiv.org/abs/2209.12827 — solved within 0.5 m of a target 1–5 m away.
51. Improbable-AI/walk-these-ways `go1_gym/envs/base/curriculum.py` (master). https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/base/curriculum.py — thresholds lin 0.8, ang 0.5, contact 0.8.
52. Ntagkas, Kiourt, Chatzilygeroudis. *PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion.* arXiv:2510.18348, 2025. https://arxiv.org/html/2510.18348 — p = 0.65; n_eval agents; U[−1, 1]; 5 seeds × 1000 at 0.7 v_max.
53. Handa et al. *DeXtreme.* ICRA 2023; arXiv:2210.13702. https://arxiv.org/html/2210.13702 — Sec. 2.6: 40 % eval envs; N = 256; t_H 20 / t_L 5; queue cleared.
54. Zhang, Rudin, Hoeller, Hutter. *Learning Agile Locomotion on Risky Terrains.* IROS 2024; arXiv:2311.10484. https://arxiv.org/html/2311.10484 — demote only when not better than random actions.
55. Kim, Oh, Lee, Choi, Ji, Jung, Youm, Hwangbo. *Not Only Rewards But Also Constraints.* IEEE T-RO 40, 2024; doi:10.1109/TRO.2024.3400935. https://arxiv.org/html/2308.12517 — Sec. V-D promote-only stage; uniform sampling; forgetting prevented.
56. Wang, Wang, Ren, Ben, Huang, Zhang, Pang. *BeamDojo.* RSS 2025; arXiv:2502.10363. https://arxiv.org/html/2502.10363v3 — three consecutive crossings; no demotion before all levels passed.
57. Portelas, Colas, Hofmann, Oudeyer. *Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments.* CoRL 2019; arXiv:1910.07224. https://arxiv.org/abs/1910.07224 — ALP forgetting signal; 20 % random.
58. Tzannetos, Gomes Ribeiro, Mandal, Singla. *Proximal Curriculum for RL Agents.* TMLR 2023; arXiv:2304.12877. https://arxiv.org/html/2304.12877 — rollout estimates cost extra steps; critic-value proxy.
59. Lai, Cao, Xu, Wu, Lin, Kong, Yu, Zhang. *World Model-based Perception for Visual Legged Locomotion.* arXiv:2409.16784, 2024. https://arxiv.org/html/2409.16784 — Table I returns.
60. Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* RSS 2021; arXiv:2107.04034. https://ar5iv.labs.arxiv.org/html/2107.04034 — proprioceptive only; sand, mud, memory foam.
61. Zhu, Huang, Mou, Zhao. *Robust Robot Walker: Learning Agile Locomotion over Tiny Traps.* arXiv:2409.07409, 2024. https://arxiv.org/html/2409.07409v2 — pits 0.05–0.2 m (trained to 0.30); circle layout; 8 s; Mix 0.934 / 0.000; pits 0.902 / 0.796 / 0.000.
62. Roennau, Heppner, Nowicki, Zoellner, Dillmann. *Reactive posture behaviors for stable legged locomotion over steep inclines and large obstacles.* IROS 2014; doi:10.1109/IROS.2014.6943257. https://api.openalex.org/works/doi:10.1109%2FIROS.2014.6943257 — abstract only.
63. Bjelonic, Kottege, Homberger, Borges, Beckerle, Chli. *Weaver: Hexapod robot for autonomous navigation on unstructured terrain.* J. Field Robotics 35(7), 2018; doi:10.1002/rob.21795. https://research.csiro.au/robotics/paper-weaver-hexapod-robot-for-autonomous-navigation-on-unstructured-terrain — roughness and step-height representation; gait adaptation.
64. Isaac Lab `velocity/config/spot/flat_env_cfg.py` (b0542fe2). https://raw.githubusercontent.com/isaac-sim/IsaacLab/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py — cobblestone 0.02–0.05 m; height_scanner None.
65. Jiang, Liang, Li, Ding, Zhu. *Omnidirectional Humanoid Locomotion on Stairs via Unsafe Stepping Penalty and Sparse LiDAR Elevation Mapping.* arXiv:2603.07928, 2026. https://arxiv.org/html/2603.07928 — 1.4 × 1.0 m at 0.05 m; hazard; safe-stepping rate; level 6 plateau.
66. He, Zhang, Jenelten, Grandia, Bächer, Hutter. *Attention-Based Map Encoding for Learning Generalized Legged Locomotion.* arXiv:2506.09588, 2025. https://arxiv.org/html/2506.09588v1 — 18 000 / 3600 epochs; entropy 0.005 → 0.002; C2 worst; 10 cm example; success / failed / stuck.
67. Fan, Flynn, Xu, He, Cramariuc, Hutter, Baines. *Learning Locomotion on Discrete Terrain via Minimal Proximity Sensing.* arXiv:2606.31912, 2026. https://arxiv.org/html/2606.31912v2 — aggregated map 51.7 → 24.3 %.
68. Yang, Yang, Wang. *Neural Volumetric Memory for Visual Locomotion Control.* CVPR 2023; arXiv:2304.01201. https://arxiv.org/html/2304.01201v1 — front but not rear legs; stones 14.4 vs 0.9.
69. Schilling, Hoinville, Schmitz, Cruse. *Walknet, a bio-inspired controller for hexapod walking.* Biological Cybernetics 2013. https://pmc.ncbi.nlm.nih.gov/articles/PMC3755227 — searching movements.
70. Bläsing, Cruse. *Stick insect locomotion in a complex environment: climbing over large gaps.* J Exp Biol 207:1273–1286, 2004; doi:10.1242/jeb.00888. https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1242/jeb.00888&resultType=core&format=json — front-leg step into the gap reduces stance velocity.
71. Bläsing, Cruse. *Mechanisms of stick insect locomotion in a gap-crossing paradigm.* J Comp Physiol A 190:173–183, 2004; doi:10.1007/s00359-003-0482-3. https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1007/s00359-003-0482-3&resultType=core&format=json — gaps > 2× step; slowing, searching, short steps.
72. Fukuhara, Suda, Kano, Kobayashi, Ishiguro. *Adaptive Interlimb Coordination Mechanism for Hexapod Locomotion Based on Active Load Sensing.* Frontiers in Neurorobotics 16, 2022. https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.645683/full — < 30 % → ≈ 60 %; CoM leaves the polygon.
73. Gîlcă. *Studying walking hexapods robots.* Annals of Constantin Brâncuși Univ. Târgu Jiu, 2021. https://www.utgjiu.ro/rev_mec/mecanica/pdf/2021-01/23_Gheorghe%20G%C3%8ELC%C4%82%20-%20STUDYING%20WALKING%20HEXAPODS%20ROBOTS.pdf — λ ≥ 3/n; tripod λ 1/2; wave 2/3, 5/6.
74. Yang, Kim. *Fault-tolerant locomotion of the hexapod robot.* IEEE Trans. SMC-B 28(1), 1998. https://research.knu.ac.kr/en/publications/fault-tolerant-locomotion-of-the-hexapod-robot-2/ — fault stability margin.
75. Yang. *Foot force distribution of a hexapod robot with fault-tolerant tripod gait.* J. ICROS 2009. https://koreascience.kr/article/JAKO200919061728002.pdf — less margin than normal gaits.
76. Ren et al. *VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception.* arXiv:2502.14814, 2025. https://arxiv.org/html/2502.14814v2 — gap curriculum; blind policy with 10 % noise and ≤ 0.5 s delay; Table II 84.81 vs 48.71 % at 100 % noise.
77. Siekmann, Green, Warila, Fern, Hurst. *Blind Bipedal Stair Traversal via Sim-to-Real RL.* RSS 2021; arXiv:2105.08328. https://ar5iv.arxiv.org/html/2105.08328 — 10–21 cm; proprioceptive reflexes; 10/10 descent.
78. Isaac Lab `velocity_env_cfg.py` TerminationsCfg (v2.2.1). https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.2.1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py — time_out and base_contact only.
79. Yang, Yuan, Zhu, Yu, Li. *Multi-expert learning of adaptive legged locomotion.* Science Robotics 5(49), 2020; arXiv:2012.05810. https://arxiv.org/pdf/2012.05810 — 90° termination; recovery on the time limit only.
80. Chane-Sane, Leziart, Flayols, Stasse, Souères, Mansard. *CaT: Constraints as Terminations for Legged Locomotion RL.* IROS 2024; arXiv:2403.18765. https://arxiv.org/html/2403.18765v1 — ET-MDP and hard p_max fail.
81. Cheng, Shi, Agarwal, Pathak. *Extreme Parkour with Legged Robots.* ICRA 2024; arXiv:2309.14341. https://arxiv.org/html/2309.14341v1 — Eq. 3 5 cm edge term; Table 2 MEV 0.03 vs 0.08.
82. Jenelten, Grandia, Farshidian, Hutter. *TAMOLS: Terrain-Aware Motion Optimization for Legged Systems.* IEEE T-RO 2022; arXiv:2206.14049. https://arxiv.org/pdf/2206.14049v2 — Sec. V-B.5 gradient penalty; blur width.
83. Kamohara, Wu, Zong, Goldman, Nakka, Hutchinson, Zhao. *GM-Loco: Terrain-Adaptive Humanoid Locomotion on Granular Media.* arXiv:2609.10286, 2026. https://arxiv.org/html/2609.10286 — Warp RFT; 4096 agents, one RTX 4090; Table II 98/0/0 vs 100/100/100.
84. Farook Deen. *Investigating Robot Learning of Quadrupedal Locomotion on Deformable Terrain* (M.Sc. thesis project page, RWTH Aachen, 2025). https://fmdazhar.github.io/projects/3_project — PBD "entirely on the CPU"; GPU memory/throughput limits.
85. Kottege, Parkinson, Moghadam, Elfes, Singh. *Energetics-informed hexapod gait transitions across terrains.* ICRA 2015; doi:10.1109/ICRA.2015.7139915. https://manuscript.isc.ac/Inventory/49/716775.htm — abstract.
86. Irawan, Nonami. *Compliant Walking Control for Hydraulic Driven Hexapod Robot on Rough Terrain.* JRM 23(1):149–162, 2011. https://doi.org/10.20965/jrm.2011.p0149 — hard bricks and rubber pads.
87. Rudin, He, Aurand, Hutter. *Parkour in the Wild.* arXiv:2505.11164, 2025. https://arxiv.org/html/2505.11164v1 — Sec. 2.3; Table 4; 3 % new terrain; 1000 roll-outs at 90 %.
88. leggedrobotics/rsl_rl `on_policy_runner.py` (main). https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/runners/on_policy_runner.py — save(): alg state, iter, infos.
89. Berner et al. (OpenAI). *Dota 2 with Large Scale Deep RL.* arXiv:1912.06680, 2019. https://arxiv.org/abs/1912.06680 — Sec. 3.3 and App. B surgery; Sec. 4.2 Rerun.
90. Chen, Goodfellow, Shlens. *Net2Net.* ICLR 2016; arXiv:1511.05641. https://arxiv.org/abs/1511.05641 — function-preserving growth.
91. Rusu et al. *Progressive Neural Networks.* arXiv:1606.04671, 2016. https://arxiv.org/abs/1606.04671 — "immune to forgetting".
92. Schmitt, Hudson, Zidek et al. *Kickstarting Deep RL.* arXiv:1803.03835, 2018. https://arxiv.org/abs/1803.03835 — ≈ 10× fewer steps; + 42 %.
93. Zhuang, Fu, Wang et al. *Robot Parkour Learning.* CoRL 2023; arXiv:2309.05665. https://arxiv.org/abs/2309.05665 — Table 2 retention.
94. Chane-Sane, Amigo, Flayols, Righetti, Mansard. *SoloParkour.* CoRL 2024; arXiv:2409.13678. https://arxiv.org/html/2409.13678v1 — beats DAgger / BC at equal compute; < 20 h on RTX 4090.
95. Zhang et al. *Distillation-PPO.* arXiv:2503.08299, 2025. https://arxiv.org/html/2503.08299v1 — 0.5 / 0.5 loss; qualitative.
96. Ramrakhya, Batra, Wijmans, Das. *PIRLNav.* CVPR 2023; arXiv:2301.07302. https://arxiv.org/abs/2301.07302 — critic-only phase; + 1.6 → + 9.9 %.
97. Wolczyk, Cupiał, Ostaszewski et al. *Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem.* ICML 2024; arXiv:2402.02868. https://arxiv.org/html/2402.02868v3 — BC best; KS fails; actor only.
98. *PPF: Pre-training and Preservative Fine-tuning of Humanoid Locomotion via Model-Assumption-based Regularization.* arXiv:2504.09833, 2025. https://arxiv.org/html/2504.09833v2 — Table II 18.9 / 28.7 / 30.7.
99. *Imitating and Finetuning Model Predictive Control for Robust and Symmetric Quadrupedal Locomotion.* arXiv:2311.02304, 2023. https://arxiv.org/html/2311.02304 — std 1.0 (2.0), LR 1e-5, clip 0.05 / 0.1; vanilla std 3.0.
100. Andrychowicz, Raichuk, Stańczyk et al. *What Matters In On-Policy RL?* ICLR 2021; arXiv:2006.05990. https://arxiv.org/abs/2006.05990 — initial std 0.5 best except Hopper.
101. Luo, Li, Yu, Wang, Wu, Zhu. *PIE: Parkour with Implicit-Explicit Learning.* IEEE RA-L 2024; arXiv:2408.13740. https://arxiv.org/html/2408.13740v3 — ten levels; 40 × 100 robots.
102. Hirose, Tsukagoshi, Yoneda. *Static Stability Criterion for Walking Robots on Irregular Terrains.* JRSJ 16(8):1076–1082, 1998; doi:10.7210/jrsj.16.1076. https://www.jstage.jst.go.jp/article/jrsj1983/16/8/16_8_1076/_article/-char/en — NESM.
103. Hirose, Lazarenko, Endo. *Directional Normalized Energy Stability Margin.* 2nd IFToMM Asian Conf. MMS, 2012. https://t2r2.star.titech.ac.jp/rrws/file/CTT100645736/ATD100000413/ — S_NESM = S_ESM/mg; slope experiments (background).
104. Papadopoulos, Rey. *The force-angle measure of tipover stability margin for mobile manipulators.* Vehicle System Dynamics 33(1):29–48, 2000. https://dspace.lib.ntua.gr/xmlui/handle/123456789/13824?show=full — CoM-height sensitivity.
105. Zhang, Zha, Guo, Chen, Sun, Wang. *Heavy-duty hexapod robot sideline tipping judgment and recovery.* Robotica 42(5):1403–1419, 2024; doi:10.1017/S0263574724000274. https://api.crossref.org/works/10.1017%2FS0263574724000274 — FASM tipping; recovery foothold (abstract).
106. Černý, Čížek, Faigl. *On Evaluation of Motion Gaits Energy Efficiency with a Hexapod Crawling Robot.* Acta Polytechnica CTU Proc. 6:6–10, 2016; doi:10.14311/APP.2016.6.0006. https://ojs.cvut.cz/ojs/index.php/APP/article/view/3925 — only adaptive gaits on wooden blocks.
107. Krabby project. *Construction-site omni teacher (closed, not accepted).* `crab_hex_omni_task/experiments/2026-10-28_site_teacher/REPORT.md` (gate table; "Why gate b failed"; "How episodes fail"; low band; Budget; Lessons 1–8; Disclosures 14), `DESIGN.md` §4.2–4.3 (v1 / v2 layouts), `fk_reach.json` (reach caps); worktree `sen0581-cameras`.
108. Krabby project. *Rod-speed DR in teacher training.* `crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr/REPORT.md` (L9 TDR; L133 1.22 h; L135 student 9.6 h; L315 field_hard 0.87 → 0.96), `CHARTER.md` L406 (std 0.80); worktree `dr-stress-sweep`.
109. Krabby project. `crab_hex_omni_task/experiments/lit-review-construction-terrain-dr.md` (2026-10-28; §1 holes row and bump probe; §5 scatter-pit; §6 scan pitch; §8 soft ground; §11 promote knob).
110. Krabby project. `crab_hex_omni_task/experiments/lit-review-obstacle-avoidance.md` (2026-10-29).
111. Krabby project. `crab_hex_omni_task/experiments/lit-review-domain-randomization.md` (2026-11-06; worktree `dr-stress-sweep`).

## Appendix: unverified or dropped

- **VB-Com "a blind policy matched the perceptive one (83.76 vs 84.05 %)".** Dropped as worded. 84.05 % is the vision-blind composite at 0 % noise; the vision-only policy scored 73.57 %, below blind. The reference was marked as not supporting the claim. Only the v2 Table II comparison at 100 % noise (84.81 vs 48.71 %) is cited [76]. The vision-only drop from 73.6 to 48.7 % and the blind policy's "unnecessary contact" on hardware were confirmed in substance but are not cited from the failing reference.
- **RLOC (Gangapurwala et al., T-RO 38(5), 2022; https://ar5iv.labs.arxiv.org/html/2012.03094).** Dropped. Its footstep planner was trained with SAC, not PPO, and only the penalties (not the stability margin) were ramped.
- **Corrected sweep wording used above.**
  - C1: stock commands are symmetric, not forward-biased; distance is measured from the env origin; Robot Parkour Learning's time-out gate is conditional and applies to the generic branch only.
  - C2: Lee's label does not measure turning; only LP-ACRL runs in Isaac Lab; WTW is a command-bin curriculum in Isaac Gym.
  - C4: old-terrain drift is up to 0.6 points, not 0.3.
  - C5: the ratio cluster widens to 0.67 (B2 is rated > 45°; the CPG hexapod gives no height).
  - C6: tan θ ≤ μ is our derivation; MULE's gain being "largest on stairs" is unsupported; "Barry only on flat ground" was too strong; the > 300 kg paper claims "first", not "only".
  - C8: loose ground rests on RMA demonstrations; Robust Robot Walker's 0 holds on Mix only (0.279 on Pole); Lee's saliency is high from the foot-trapping event onward.
  - C9: He et al. give a single 10 cm example; their per-terrain map drift is not tied to stage 2; Fan's foot-sensor result is qualitative.
  - C11: the thesis attributes the CPU bottleneck to frame rate and the scaling limit to GPU memory and throughput; the page is internally inconsistent (it also mentions a 6 × 6 m pit and 20 mm spheres).
  - C12: Černý's stability ranking is qualitative; NESM's slope result is background cited in the 2012 paper.
- **Weaver's 30° inclines** come from the CSIRO project page, where they are credited to the proprioceptive controller (CT [10]); the paper page opened here supports only the stereo-derived descriptors [63].
- **Verified but not used, for length:** DreamRiser (arXiv 2306.12712; fall recovery, no holes); Beyond Robustness (arXiv 2507.07825; qualitative only); Dürr 2001 (kinematic numbers paywalled); the cyberneticzoo Ambler page (heights 3.5 vs 5 m disagree); McGhee & Waldron 1985; Hodoshima et al. 2007; JPL ATHLETE; Stuart et al. 1992.
- **Venue labels not confirmed on the opened pages:** Fan et al. as IROS 2026 [67]; MULE is under review [23]; the heavy hydraulic paper is RA-L per its arXiv journal-ref [25]; Kim et al. is T-RO per DOI [55].
- **In-repo discrepancies.**
  - The brief's "TDR field_hard ≈ 0.89": the rod-speed REPORT's no-regression block reads TDR 0.96 against T\* 0.87 (L315). The site step-9 read for T\* was 0.8875 (es1–4).
  - "TDR fails real holes (0.11, 89 / 100)" is T\*'s holes read (CT §1); no TDR holes row was found.
  - The ledge_down d 0.3–0.6 metres (≈ 0.125–0.23 m) come from the brief and are consistent with the b4 ladder (level 2 = 0.098 m; primary bottom 0.23 m at level 5.4).
  - std is 0.81 in the site report and 0.80 in the TDR CHARTER.
- **Inferences, not source statements:**
  - the blind threshold ≈ 0.12–0.14 m (0.3 × the 0.46 m drop reach, against the ledger's 0.98 → 0.29 step);
  - tan θ ≤ 0.7 μ as a per-env sampling constraint;
  - every top-rung value in recommendation 3, and the 0.6 m pothole-width cap;
  - the 20–30 % retention share;
  - scoring stand slots by survival plus bounded drift;
  - std ≈ 0.5;
  - the ≈ 1.0 rad pitch limit, and that the 0.5 rad termination deletes the states recovery needs;
  - support-failure events as a loose-ground stand-in;
  - that v2 surgery requires exact v1 semantics;
  - the mapping of Fukuhara's failure onto Krabby.
- **Unrefereed sources, used with that caveat:**
  - arXiv preprints: [8, 9, 13–15, 23, 52, 59, 61, 65–67, 76, 83, 87, 89, 91, 92, 95, 98, 99] (Robot Parkour Learning [93] is CoRL 2023);
  - code and docs: [1–5, 41–49, 51, 64, 78, 88];
  - web pages, slides and press: [10–12, 22, 26–30, 34, 37];
  - abstract-only reads: [36, 38, 62, 85, 105];
  - thesis page: [84].
