73. Puny, Atzmon, Ben-Hamu, Misra, Grover, Smith, Lipman. *Frame Averaging for Invariant and Equivariant Network Design.* ICLR 2022; arXiv:2110.03336. https://arxiv.org/abs/2110.03336 — exact equivariance preserving backbone expressivity; supervised only.
74. Lan, Wang, Li, Jiang, Fu, Su, Wong, Jin, Wang. *SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour.* arXiv:2606.19928, 2026. https://arxiv.org/html/2606.19928 — 5-step history mirrored pointwise per element; one-sided terrain transfer.
75. Nie, Zhang, Jin, Cao, Lin, Yang, Gao. *Coordinated Humanoid Robot Locomotion with Symmetry Equivariant RL Policy (SE-Policy).* AAAI 2026; arXiv:2508.01247. https://arxiv.org/html/2508.01247 — joint symmetric command sampling; equivariant actor / invariant critic; no seed count.
76. Zhen, Zhang, Wang, Yu. *Reflex: RL with Reflection Symmetry Exploitation in State-Based Continuous Control.* arXiv:2605.23415, 2026. https://arxiv.org/html/2605.23415 — policy + critic symmetry regularisers, w_t = w0(1 − t/T); 5 seeds; RAD baselines only.
77. Huang, Zhu, Du, Zhao. *MoE-Loco: Mixture of Experts for Multitask Locomotion.* IROS 2025; arXiv:2503.08564. https://arxiv.org/abs/2503.08564 — Table III gradient cosine −0.132 (MLP) vs 0.278 (MoE), quadruped–biped pairs only.
78. Yu, Kumar, Gupta, Levine, Hausman, Finn. *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS 2020; arXiv:2001.06782. https://arxiv.org/abs/2001.06782
79. Liu, Liu, Jin, Stone, Liu. *Conflict-Averse Gradient Descent for Multi-task Learning (CAGrad).* NeurIPS 2021; arXiv:2110.14048. https://arxiv.org/abs/2110.14048
80. Kurin, De Palma, Kostrikov, Whiteson, Kumar. *In Defense of the Unitary Scalarization for Deep Multi-Task Learning.* NeurIPS 2022; arXiv:2201.04122. https://arxiv.org/abs/2201.04122 — summed losses + regularisation match multi-task optimisers.
81. Rudin, He, Aurand, Hutter. *Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-expert Distillation and RL Fine-tuning.* arXiv:2505.11164, 2025 (IJRR 2026, doi 10.1177/02783649261455067). https://arxiv.org/abs/2505.11164 — Sec. 2.2 DAgger loss + action noise; Sec. 2.3 single-mode collapse; fine-tuning instability, critic pre-training with frozen policy, reduced std.
82. van der Pol, Worrall, van Hoof, Oliehoek, Welling. *MDP Homomorphic Networks.* NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/2be5f9c2e3620eb73c2972d7552b6cb5-Abstract.html — CartPole, grid world, Pong only.
83. Wang, Zhong, Chang, Allen-Blanchette. *Local-Canonicalization Equivariant GNNs for Swarm Robot Control (LEGO).* IROS 2026; arXiv:2509.14431. https://arxiv.org/abs/2509.14431 — MARL/drones, no locomotion.
84. Isaac Lab ANYmal-D `rsl_rl_ppo_cfg.py` (`AnymalD{Flat,Rough}PPORunnerWithSymmetryCfg`), `isaaclab_rl/rsl_rl/symmetry_cfg.py`, `mdp/symmetry/anymal.py` (main; local clone b0542fe2d). https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py — augmentation opt-in over 4 images; mirror_loss_coeff 0.0; no config uses the loss.
85. Bao, Li, As, Krause, Hutter. *Symmetry-Guided Memory Augmentation for Efficient Locomotion Learning (SGMA).* arXiv:2502.01521, 2025. https://arxiv.org/abs/2502.01521 — Sec. IV RNN hidden-state augmentation; Sec. V naive feedforward augmentation hurts under partial observability.
86. Yu, Turk, Liu. *Learning Symmetric and Low-Energy Locomotion.* ACM TOG (SIGGRAPH) 2018; arXiv:1801.08093. https://arxiv.org/abs/1801.08093 — Eq. 6–7 sum over 20,000 samples, w = 4; hexapod in Fig. 1c/6.
87. Seo, Sferrazza, Geng, Nauman, Yin, Abbeel. *FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control.* arXiv:2505.22642, 2025. https://arxiv.org/abs/2505.22642 — < 3 h on one A100; batch 32,768; GPU buffer; Sec. 3 reward/algorithm gait coupling; 48-hour HumanoidBench baselines.
88. Seo et al. FastTD3 repository (README, `hyperparams.py`, `fast_td3.py`, `fast_td3_utils.py`). https://github.com/younggyoseo/FastTD3 — A100 80 GB defaults; IsaacLab 4096 envs, buffer 1024×10; nn.Tanh actor; float32 obs/next_obs buffers.
89. Raffin. *Getting SAC to Work on a Massive Parallel Simulator: Tuning for Speed (Part II).* Blog, Jul 2025. https://araffin.github.io/post/tune-sac-isaac-sim/ — 1024 envs; 2.5–97.5th-percentile bounds; Optuna CMA-ES; γ 0.983; "slight performance gap".
90. Li, Chen, Hong, Ajay, Agrawal. *Parallel Q-Learning: Scaling Off-policy RL under Massively Parallel Simulation.* ICML 2023, PMLR 202; arXiv:2307.12983. https://arxiv.org/abs/2307.12983 — Sec. 4.4.1 env-count sensitivity; 5 seeds; three processes.
91. Hafner, Pasukonis, Ba, Lillicrap. *Mastering diverse control tasks through world models (DreamerV3).* Nature 640:647–653, 2025; arXiv:2301.04104. https://arxiv.org/abs/2301.04104 — Proprio Control 500K steps; 16 env instances; one A100; fixed-HP Acme PPO.
92. Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024; arXiv:2310.16828. https://arxiv.org/abs/2310.16828 — 14M steps; UTD 1 vs 512; Dog instabilities; 33 GPU-days for the 317M multitask model.
93. Bhatt, Palenicek, Belousov, Argus, Amiranashvili, Brox, Peters. *CrossQ.* ICLR 2024; arXiv:1902.05605. https://arxiv.org/abs/1902.05605 — UTD 1; single env; 5M steps.
94. Nauman, Ostaszewski, Jankowski, Miłoś, Cygan. *Bigger, Regularized, Optimistic (BRO).* NeurIPS 2024; arXiv:2405.16158. https://arxiv.org/abs/2405.16158 — within 1M steps; memory limitation.
95. Lee, Hwang, Kim, Kim, Tai et al. *SimBa: Simplicity Bias for Scaling Up Parameters in Deep RL.* ICLR 2025; arXiv:2410.09754. https://arxiv.org/abs/2410.09754 — PPO + SimBa at 1024 envs on Craftax (whole block).
96. Gallici, Fellows, Ellis, Pou et al. *Simplifying Deep Temporal Difference Learning (PQN).* ICLR 2025; arXiv:2407.04811. https://arxiv.org/abs/2407.04811 — App. A.3 deterministic policy for discrete actions.
97. Yi, Choi, Singh, Huang, Truong, Sferrazza, Ma, Duan, Abbeel, Shi, Liu, Kanazawa. *Flow Policy Gradients for Robot Control (FPO++).* arXiv:2602.02481, 2026. https://arxiv.org/abs/2602.02481 — Fig. 8 (2⁸–2¹² envs, 5 seeds); Fig. 9 pronk vs trot at 4096 envs; 19 vs 23 min; App. C.1 baseline.
98. McAllister, Ge, Yi, Kim, Weber, Choi, Feng, Kanazawa. *Flow Matching Policy Gradients (FPO).* ICLR 2026; arXiv:2507.21053. https://arxiv.org/abs/2507.21053
99. Wang, Su, Gupta, Pathak. *Evolutionary Policy Optimization.* arXiv:2503.19037, 2025 (preprint). https://arxiv.org/abs/2503.19037 — Table 1 ANYmal / A1 parkour; ~24 GB; 24,576 envs; no sim-to-real.
100. Xie, Zhang, Yang, Hutter, Xu. *Simple Policy Optimization.* arXiv:2401.16025 v9, 2025. https://arxiv.org/abs/2401.16025 — objective; Table 1 ratio deviation 548.060 vs 0.190; 3-layer results below PPO on Ant/HalfCheetah.
101. Henderson, Islam, Bachman, Pineau, Precup, Meger. *Deep Reinforcement Learning that Matters.* AAAI 2018; arXiv:1709.06560. https://arxiv.org/abs/1709.06560 — Fig. 5 t = −9.0916, p = 0.0016.
102. Colas, Sigaud, Oudeyer. *How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments.* arXiv:1806.08295, 2018. https://arxiv.org/abs/1806.08295 — β = 0.51 at N = 5; bootstrap < 10–20.
103. Colas, Sigaud, Oudeyer. *A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms.* arXiv:1904.06979, 2019. https://arxiv.org/abs/1904.06979 — Sec. 4.5 N vs ε; Welch at N < 5; bootstrap < 50; bimodal inflation.
104. Agarwal, Schwarzer, Castro, Courville, Bellemare. *Deep RL at the Edge of the Statistical Precipice.* NeurIPS 2021; arXiv:2108.13264. https://arxiv.org/abs/2108.13264 — IQM; stratified bootstrap; N = 10 across-task CIs.
105. Patterson, Neumann, White, White. *Empirical Design in Reinforcement Learning.* JMLR 25(318), 2024. https://www.jmlr.org/papers/v25/23-0183.html — Sec. 2 summary item 5; Sec. 3.2 maximisation bias.
106. Eimer, Lindauer, Raileanu. *Hyperparameters in RL and How To Tune Them.* ICML 2023, PMLR 202; arXiv:2306.01324. https://arxiv.org/abs/2306.01324 — unstable top seeds; 3 × 5 tuning / 10 test seeds; 1/12 budget.
107. Bjorck, Gomes, Weinberger. *Is High Variance Unavoidable in RL? A Case Study in Continuous Control.* ICLR 2022; arXiv:2110.11222. https://arxiv.org/abs/2110.11222 — early outlier runs; saturating nonlinearities; penultimate normalisation (DrQ-v2).
108. Jordan, Chandak, Cohen, Zhang, Thomas. *Evaluating the Performance of RL Algorithms.* ICML 2020; arXiv:2006.16958. https://arxiv.org/abs/2006.16958
109. Isaac Lab. *Reproducibility and Determinism.* https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html — identical under fixed hardware/version; runtime writes reorder; LSB divergence.
110. Nagarajan, Warnell, Stone. *Deterministic Implementations for Reproducibility in Deep RL.* arXiv:1809.05676, 2018. https://arxiv.org/abs/1809.05676
111. Mania, Guy, Recht. *Simple random search provides a competitive approach to RL (ARS).* NeurIPS 2018; arXiv:1803.07055. https://arxiv.org/abs/1803.07055 — Sec. 4.5 / Fig. 3, 100 seeds, 70 % (Walker2d 20 %).
112. Petrenko, Allshire, State, Handa, Makoviychuk. *DexPBT.* RSS 2023; arXiv:2305.12127. https://arxiv.org/abs/2305.12127 — Sec. IV one V100 16 GB per agent, 8192 envs; PBT after 2e8 steps; smaller updates late.
113. Shahid, Narang, Petrone, Ferrentino, Handa, Fox, Pavone, Roveda. *Benchmarking Population-Based RL across Robotic Tasks with GPU-Accelerated Simulation.* IEEE CASE 2025; arXiv:2404.03336; code README. https://arxiv.org/abs/2404.03336 ; https://github.com/Asad-Shahid/PBRL — one RTX 4090; 1024 envs per agent; "num_envs must divide num_agents"; off-policy capped at 8.
114. Isaac Lab. *Population Based Training.* https://isaac-sim.github.io/IsaacLab/main/source/features/population_based_training.html — rl_games only; one process per policy; 6+ workers; mutation keys.
115. Cully, Clune, Tarapore, Mouret. *Robots that can adapt like animals.* Nature 521, 2015 (doi:10.1038/nature14422); arXiv:1407.3501. https://arxiv.org/abs/1407.3501 — ~13,000 gaits; 6-D duty-factor map; 40M evaluations; < 2 min adaptation.
116. Batra, Tjanaka, Fontaine, Petrenko, Nikolaidis, Sukhatme. *PPGA.* ICLR 2024; arXiv:2305.13795. https://arxiv.org/abs/2305.13795 — 4× best reward on humanoid.
117. Zahavy et al. *Discovering Policies with DOMiNO.* ICLR 2023; arXiv:2205.13521. https://arxiv.org/abs/2205.13521
118. Margolis, Agrawal. *Walk These Ways.* CoRL 2022, PMLR 205; arXiv:2212.03238. https://arxiv.org/abs/2212.03238 — command-conditioned family (MoB).
119. Iscen, Caluwaerts, Tan, Zhang, Coumans, Sindhwani, Vanhoucke. *Policies Modulating Trajectory Generators (PMTG).* CoRL 2018; arXiv:1910.02812. https://arxiv.org/abs/1910.02812 — Sec. 4.3: 77-parameter linear policy; without PMTG both fall short of optimal reward.
120. Yang, Gao. *Terrain-adaptive Central Pattern Generators with RL for Hexapod Locomotion.* arXiv:2310.07744, 2023. https://arxiv.org/abs/2310.07744 — Sec. IV, Fig. 4 (4 runs): CPG modulation beats foot-position output.
121. Schilling, Konen, Ohl, Korthals. *Decentralized Deep RL for a Distributed and Adaptive Locomotion Controller of a Hexapod Robot.* IROS 2020; arXiv:2005.11164. https://arxiv.org/abs/2005.11164 — 15 seeds; 657.11 (SD 68.07) vs 546.70 (SD 131.23), p = .011; Gazebo/Bullet PhantomX.
122. Seyde, Werner, Schwarting, Gilitschenski, Riedmiller, Rus, Wulfmeier. *Solving Continuous Control via Q-learning (DecQN).* ICLR 2023; arXiv:2210.12566. https://arxiv.org/abs/2210.12566 — 10 seeds; Mini Cheetah qualitative (Fig. 8).
123. Zhou, Vanschoren. *Open-Ended Learning Strategies for Learning Complex Locomotion Skills.* 5th Workshop on Meta-Learning at NeurIPS 2021; arXiv:2206.06796. https://arxiv.org/abs/2206.06796 — Table 1 PPO 36 %, SAC 43 %, ePOET 64 %; different curricula; no seed repeats.
124. NVIDIA Developer Forums thread 268134 (Sep–Nov 2023). https://forums.developer.nvidia.com/t/multiple-isaac-sim-containers-on-one-gpu-fails-with-cuda-illegal-memory-access-in-omni-physx-tensors-plugin/268134 — illegal memory access in `get_jacobians`; unresolved.
125. isaac-sim/IsaacLab issue #4240 (Dec 2025). https://github.com/isaac-sim/IsaacLab/issues/4240 — "only 2 jobs could be ran even I have plenty of VRAM left"; no maintainer answer.
126. NVIDIA Developer Forums thread 331011 (Apr–May 2025, Isaac Sim 4.5). https://forums.developer.nvidia.com/t/trouble-running-multiple-isaac-sim-processes/331011 — 3rd/4th processes hang on shader compilation; Kit "is not designed to run like this".
127. isaac-sim/IsaacLab discussion #1241 / issue #877 (2024). https://github.com/isaac-sim/IsaacLab/discussions/1241 — 80 s → ~500 s per item under multiprocessing; Ray suggested.
128. NVIDIA. *Multi-Process Service* (Release 615). https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf — Sec. 1.1.2.1 when to use; Linux only; fault containment; signal termination undefined.
129. Wang, Feng, Wang, Geng, Li, Ding. *GMI-DRL: Empowering Multi-GPU Deep RL with GPU Spatial Multiplexing / Adaptive-Grained Parallelism.* arXiv:2206.08482 (2022); USENIX ATC 2025. https://arxiv.org/abs/2206.08482 ; https://www.usenix.org/system/files/atc25-wang-yuke.pdf — Isaac Gym PPO utilisation "under 50 %"; MPS/MIG beat direct sharing on A100.
130. NVIDIA IsaacGymEnvs `docs/pbt.md`. https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/pbt.md — `--experiments_per_gpu=4 --num_gpus=1`.
131. Isaac Lab. *Ray Job Dispatch and Tuning.* https://isaac-sim.github.io/IsaacLab/main/source/features/ray.html — GPU split across workers; "[WARNING]: Splitting node into more than one worker".
132. PyTorch. *Model Ensembling* (`torch.func.stack_module_state` + `vmap`). https://docs.pytorch.org/tutorials/intermediate/ensembling.html — 1.33 ms looped vs 645.64 µs vectorised.
133. Cobbe, Hilton, Klimov, Schulman. *Phasic Policy Gradient.* ICML 2021; arXiv:2009.04416. https://arxiv.org/abs/2009.04416 — separate networks; value function "more aggressively optimized with a higher level of sample reuse" (Procgen).
134. van Hasselt, Guez, Hessel, Mnih, Silver. *Learning values across many orders of magnitude (PopArt).* NeurIPS 2016; arXiv:1602.07714. https://arxiv.org/abs/1602.07714
135. Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO).* NeurIPS 2022 D&B; arXiv:2103.01955. https://arxiv.org/abs/2103.01955 — Sec. 5.1 value normalisation "never hurts".
136. Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt. *Multi-task Deep RL with PopArt.* AAAI 2019; arXiv:1809.04474. https://arxiv.org/abs/1809.04474 — updates depend linearly on return scale.
137. Schaul, Ostrovski, Kemaev, Borsa. *Return-based Scaling: Yet Another Normalisation Trick for Deep RL.* arXiv:2105.05347, 2021. https://arxiv.org/abs/2105.05347
138. Isaac Lab `isaaclab/managers/reward_manager.py`. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/managers/reward_manager.py — `compute()` accumulates weighted dt-scaled terms, no clip.
139. Schwarke, Mittal, Rudin, Hoeller, Hutter. rsl_rl `algorithms/distillation.py` (library paper arXiv:2509.10771). https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/distillation.py — raw `teacher(obs).detach()` target; mse/huber.
140. Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* RSS 2021; arXiv:2107.04034. https://arxiv.org/abs/2107.04034 — phase 2 MSE on the latent only.
141. Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning robust perceptive locomotion for quadrupedal robots in the wild.* Science Robotics 2022; arXiv:2201.08117. https://arxiv.org/abs/2201.08117 — Lbc + 0.5 Lre.
142. Czarnecki, Pascanu, Osindero, Jayakumar, Swirszcz, Jaderberg. *Distilling Policy Distillation.* AISTATS 2019; arXiv:1902.02186. https://arxiv.org/abs/1902.02186
143. Krabby PPO review. *Evidence dossier (final): is PPO, or its hyper-parameters, limiting Krabby?* 2026-09-30. `/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/dossier/DOSSIER.md` — §0 summary; §1 effective setup; §3 telemetry; §4.1–4.7 campaign evidence; §6 constraints; §10 reading.

## Appendix: unverified or dropped items

- **Robotic World Model (Li, Krause, Hutter; arXiv:2501.10100 v5).** Exists and was opened. DROPPED from
  §10 as cited because the sweep reversed the finding: Table 1 gives RWM + MBPO-PPO a real tracking reward of
  0.90 ± 0.04 against PPO's 0.90 ± 0.03 with 6M vs 250M transitions, but 55 vs 10 min wall-clock (50 min model
  pretraining + 5 min policy); the "still falls short of ... well-tuned model-free RL" quote is from its
  general Limitations section. Read correctly it strengthens the wall-clock case against model-based RL here.
- **Salimans et al., *Evolution Strategies as a Scalable Alternative to RL* (arXiv:1703.03864).** Exists;
  DROPPED from §11 because the sweep's "3–10x TRPO's data" is the Atari-vs-A3C figure; against TRPO on MuJoCo
  the paper says "no more than 10x" and up to 3x better on simple tasks. Its Humanoid gaits "never observed
  with TRPO" still support the claim that different optimisers reach different basins.
- **Smith, Kostrikov, Levine, *A Walk in the Park* (arXiv:2208.07860; RSS 2023).** Exists; DROPPED from §10
  as support for "all parallel off-policy actors bound the action inside the policy": it is a single real
  robot, and its constraint is a narrow joint-target window [p − o, p + o] (o = 0.2/0.4/0.4 rad), Sec. IV-A
  not III-A. Its "the simulated agent cannot make any progress in the unconstrained action space" remains a
  data point for bounding.
- **Cully et al. [115] as evidence that "controlled switching, not interior actions" spans the tripod
  family.** DROPPED (§4): the controller is a Gaussian-smoothed square-wave *position* CPG with interior
  amplitudes, smoothed "to remove sharp transitions"; it says nothing about bang-bang optimality. Cited only
  for MAP-Elites (§11).
- **Extreme-parkour effective clip "±2.4".** Corrected to ±4.8: the default `a1` task overrides
  action_scale to 0.25 (§2, [6]).
- **Isaac Lab entropy "0.005 in every velocity task".** Corrected: 0.005 is ANYmal C/D only; others use
  0.0025–0.01 (§2, [7]).
- **"In every reference stack the clip is non-binding relative to σ".** Corrected: false for rl_games (±1
  clip, std 1, about a third of samples truncated), which instead bounds the mean (§2, [10–12]).
- **DHAL "no seeds, confounded with multi-critic".** Corrected: 4 training seeds, 5 hardware trials per
  scenario, and the ablation is Beta-only inside the multi-critic framework (§3, [19]).
- **Su et al. "hard equivariance has large seed spread (65.7 ± 23.2 vs 54.4 ± 32.6)".** Corrected: those are
  success rates, PPOeqic's spread is the smaller, and Su attribute the one-side advantage to domain
  randomisation breaking symmetric dynamics (§8, [68]).
- **rl_games "ratchets the rate to min_lr" attributed to the legacy controller.** Corrected: the sentence
  concerns per-minibatch stepping on the step-KL signal, one seed; the legacy floor there is 5e-5 and part
  of its drift is observation-normaliser movement, which Krabby lacks (§6, [48, 41]).
- **SAPG "PPO saturation ceiling at ~24k envs".** Corrected: 24,576 is the experiment scale; the paper gives
  no ceiling (§7, [61]).
- **PPO paper "log-std init 0 for MuJoCo".** Corrected: the paper states no MuJoCo std init; Table 4
  Roboschool anneals log-std −0.7 → −1.6; std-1 is CleanRL's convention (§5, [39, 21]).
- **Hsu et al. as support for init std 0.5.** Reclassified as a counter-caveat (narrow init worsens basin
  selection) (§5, [2]).
- **He et al. occupancy 71/30/19 "for the clipped Gaussian".** Corrected: three matched tanh policies with
  the latent interface widened to ±10 (§2, [3]).
- **Geiß "+200" attributed to mean regularisation generally.** Corrected: the flipped-uniform KL objective on
  running, under GAIL with TRPO (§2, [4]).
- **Replay arithmetic "N ~ 1,024 feasible only via proprio storage".** Corrected: 256 × 1,024 × 9.2 KB ≈
  2.4 GB fits at full obs; proprio-only storage instead allows N ~ 10k in ~1.9 GB (§10, [88]).
- **VideoMimic "clip 8.0 non-binding relative to σ".** Reviewer inference (8.0 = 10 σ), not a paper
  statement (§3, [13]).
- **CAPG "adds no restoring force on |μ|".** Inference from the estimator's form; the paper does not discuss
  mean drift (§3, [20]).
- **Hard log-std clamps in SB3 SAC (`policies.py`, `th.clamp(log_std, −20, 2)`), Spinning Up SAC and skrl
  (`clip_log_std=True`, max 2).** Checked by the verification pass but carried without numbered entries; they
  establish that hard clamps are common and loose, not that a binding clamp has precedent (§5).
- **EqR (Mondal, Jain, Siddiqi, Ravanbakhsh, ICML 2022).** Located by search only, not opened; locomotion
  content unverified (§8 gap).
- **Kim et al. 2026 hexapod DRL (Actuators 15(1):33)** and other items not opened by the earlier
  gait-motion-quality review are not re-listed here.
- **Inferences, not source statements:** "lowering σ alone cannot undo Krabby's saturation" (applies He et
  al.'s P10 criterion to the 97 % figure); the (11 − 1.1)² × 18 bound-loss calibration; the α/√c mapping
  2e-4 → ~2e-5; the 69-step half-life and 17-step GAE horizon; the value-clip rate limit (~0.2 per sample per
  iteration) and its role in stage collapses; "tanh alone does not stop drift; the log-det entropy term
  does" as applied to Krabby; the cam channel as a chain of integrators with a PWM code; the Fisher-exact
  seed thresholds; "a bounded head may shrink the lottery itself"; the ordering of rungs 6–9.
- **In-repo numbers not re-measured here:** the clipped rod-mean shares (0.59–0.73 / 0.94–0.96), the
  two-process overlap (2,783 steps/s each), 4.4 GB per process, 61–69 % utilisation, and every census figure
  are quoted from the dossier [143] with its own verification tags; the dossier marks the total_terminates
  co-fire rate, the rollout/update KL mismatch (~3 % of desired_kl) and the X19 saturation trajectory as
  UNRESOLVED or indicative.
- **Unrefereed sources used with that caveat:** [3, 24, 25, 40, 60, 69, 70, 74, 76, 97, 99, 100] (2025–2026
  arXiv preprints), [9, 89] (blog), [41, 48, 49] (maintainer docs), [124–127] (forum/issue threads), [64]
  (withdrawn preprint), [123] (workshop).
