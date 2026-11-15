# Krabby Glossary

A quick-reference glossary of acronyms and terms used across the Krabby project.

For in-depth explanations of the stack see [TECHNOLOGY_AND_TERMINOLOGY.md](TECHNOLOGY_AND_TERMINOLOGY.md).

---

| Term | Expansion | What it means in Krabby |
|------|-----------|-------------------------|
| **ADC** | Analog-to-Digital Converter | The MCU circuit that turns an analog voltage (a pot wiper, an IS line) into a number 0–1023. This is what `analogRead()` returns. |
| **Bake** | — | The user's decision to make a trained head the policy of record: its stage entries are added to a task's `policy/manifest.yaml` and the files copied into `policy/` (`bundle_policy.py --sync`). |
| **c1** | — | The omni walker of record (RT_s19 `model_29500`, seed 19): TDR fine-tuned for 509 iterations on the rough-terrain mix (slopes, hills, steps up and down, potholes, gravel and rubble, plus flat and field columns; rod-speed DR kept), picked from 4 same-mix candidates by the rough-terrain campaign's short-arm selection: `rt_11_p3b_rough/model_29500.pt` (sha256 `6d1f29be`). Baked 2026-11-14 ([omni policy/README.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/README.md)). |
| **C1S** | — | The omni depth student of record: distilled from c1 (4 cameras, 5000 iterations; RT_ST_s19 `model_34499`, sha256 `6b48ef33`). `rt_12_student/` holds its ONNX export (`policy.onnx` + `depth_latest.onnx` + `omni_student_io.json`); the 122 MB `.pt` is cited by sha256, not bundled. |
| **CAL** | Calibration | Finding each joint's real min/max sensor values so a normalized command (0.0–1.0) maps onto its actual travel. |
| **Clock hide** | — | `KRABBY_OMNI_CLOCK_HIDE=1`: the gait-clock observation columns are zeroed on every frame. Every omni head of record from c6x on trained this way and must be played / evaluated with it set (see [omni policy/README.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/README.md)). |
| **Command slots (omni)** | — | The four command regimes the omni task samples: **stand**, **move** (world travel direction + speed), **turn** (new absolute facing), **move+turn** (both). See the [omni README](../parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/README.md). |
| **CLI** | Command-Line Interface | The terminal command wrapper around the SDK (e.g. `python -m firmware …`). |
| **CRC** | Cyclic Redundancy Check | A checksum stored alongside EEPROM data to detect corruption; a bad CRC means "don't trust this saved calibration." |
| **Depth student / distillation** | — | A student policy that sees depth cameras instead of the teacher's privileged height scan, trained to imitate a frozen teacher's actions (distillation). The omni student of record `rt_12_student` (C1S, 4 cameras) is distilled from c1 (the rod-speed DR pairing's `rsd_10_student`, c31000, was distilled from TDR, and F19's `f19_9_student` from T*) ([omni policy/README.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/README.md)). |
| **DoF** | Degrees of Freedom | The number of independently controllable joints. Each Krabby leg has three (yaw, hip, knee). |
| **DR** | Domain Randomization | Randomized pushes, chassis mass and CoM offsets during training (`KRABBY_DR_PUSH` / `KRABBY_DR_MASS` / `KRABBY_DR_COM`) so the policy tolerates model error; the omni walkers TDR and c1 (the walker of record) also trained with a per-env rod slew scale (`KRABBY_DR_ROD_SPEED`, omni `config/crab_hex/dr_knobs.py`; training-only, never set in play / eval / deploy). |
| **ECR** | (AWS) Elastic Container Registry | Where built locomotion container images are published. The bench watchdog polls it for new images to deploy. |
| **EEPROM** | Electrically Erasable Programmable Read-Only Memory | Tiny non-volatile memory on the MCU. Krabby stores each board's role/serial and per-joint calibration here so it survives power-off. |
| **EMI** | Electromagnetic Interference | Electrical noise (e.g. from a running motor) that a floating/unconnected sensor pin can pick up — a source of false readings the firmware has to guard against. |
| **Exposure (obstacle)** | — | Training-time telemetry of how often episodes leave the start platform and reach each obstacle (`Metrics/<term>/reach_edge_frac`, `reach_obst_frac`, `obst_coverage_k`). |
| **Friction contract (rough terrain)** | — | The one friction setup of the rough-terrain campaign, the same in training and eval (`KRABBY_RT_FRICTION=gravel:LO:HI`, `mdp/rt_friction.py`; default off; needs the site geometry with the `rt` ladder). The ground's combine mode is `multiply` at mu 1.0, so a contact's effective mu is the robot shape's own, drawn per env: non-gravel envs get mu = (b + 1) / 2 with b from TDR's 64 buckets U(0.6, 2.0) (TDR's own distribution, kept), gravel envs get U(0.2, 0.5) fixed per env (c1 trained and is evaluated on `gravel:0.2:0.5`). The privileged friction observation reads 2 mu_eff - 1 (TDR's encoding). Set on training and gravel-eval rows only, never in a play env; leaving it off on a gravel row is silent and plays rigid mu >= 0.8 ground. |
| **Gait clock** | — | A per-env phase advanced each step at a cadence linear in the commanded forward speed (frozen on stop commands); the policy observes its sin/cos and the clock rewards pay for a tripod contact schedule locked to it. |
| **HAL** | Hardware Abstraction Layer | The boundary that lets the *same* policy run against either simulation or real hardware. Talks over ZMQ; swap the backend, not the policy. |
| **Hall (sensor)** | Hall-effect sensor | A magnetic position sensor. On Krabby it gives *incremental* counts (relative motion), used on the hip-lift and yaw joints. |
| **H-bridge** | — | The motor-driver circuit (one per motor) that lets the MCU drive a motor forward or backward. Also outputs the current-sense ("IS") signal. |
| **Hexapod / "hex"** | — | A six-legged robot. Krabby's target chassis. The **training plant** is `assets/crab.usda` (generated, A15+B geometry; see [crab-hexapod-plant.md](crab-hexapod-plant.md)); `assets/crab_hex_ref.*` is the legacy teleop/HAL demo model. |
| **Head of record / policy of record** | — | The checkpoint a task ships ("current" in its `policy/manifest.yaml`) plus the stage heads it was trained through; it changes only on a bake. Superseded heads stay listed under `superseded:`. |
| **HL** | Hip-Lift | The joint that raises/lowers a leg. Driven by a Hall linear actuator. |
| **HY** | Hip-Yaw | The joint that swings a leg sideways. Uses a Hall encoder. |
| **IAM** | (AWS) Identity and Access Management | Controls *what* a device or user is allowed to do in AWS. Each bench device has its own IAM identity. |
| **IMU** | Inertial Measurement Unit | A sensor reporting orientation, angular rate, and acceleration. Tells the robot which way is up and how it's moving; sourced from the ZED camera. |
| **IS** | (German *I_S* = current sense) | The analog current-sense output from each H-bridge. Higher current ≈ more load, so `avgIS` is used as a proxy for foot contact / how hard a joint is working. |
| **Joint code** | — | 4-letter ID: 2 letters for the leg + 2 for the joint. Example: `FLHL` = Front-Left Hip-Lift. |
| **KL** | Knee-Lift | The joint that bends the knee. Uses a potentiometer. |
| **Krabby-Uno** | — | The passive v0.2 carrier *shield* that stacks on one Mega. It fans the Mega's pins out to two ribbon headers (J1, J2), one per leg = 6 motors per board. |
| **Leg prefix** | — | First two letters of a joint code: `FL FR ML MR RL RR` = Front/Middle/Rear × Left/Right. |
| **MCU** | Microcontroller Unit | The Arduino Mega 2560 that drives a set of motors. The full robot uses three Megas, one per body section (Front / Left / Right). |
| **ONNX** | Open Neural Network Exchange | A portable model format used to export the trained policy so it can run fast on the Orin (often via TensorRT). |
| **Orin** | — | NVIDIA Jetson Orin: the on-robot computer that runs inference and the HAL server. Powered by the JetPack / L4T software stack. |
| **Payload (max_stable)** | — | A random deck mass + CoM offset applied in training (`KRABBY_PAYLOAD_*`, training-only). `max_stable` is the highest payload level (kg) on the evaluated curve up to which a head is stable on every level (T*: 135 kg, sim plant). |
| **PID** | Proportional-Integral-Derivative | The closed-loop control math that moves a joint to a target position by continuously correcting the error between target and measured position. |
| **Pot** | Potentiometer | A resistive position sensor giving an *absolute* angle reading (0–1023). Used on the knee joint. |
| **PPO** | Proximal Policy Optimization | The specific RL algorithm used to train both the teacher and student policies. |
| **PWM** | Pulse-Width Modulation | How motor speed is set: the driver switches power on/off fast, and the on-fraction (duty cycle) sets effective voltage. Higher PWM = harder drive. |
| **RGB-D** | Red-Green-Blue + Depth | A camera image that carries colour *and* per-pixel distance. The depth channel is what the parkour policy "sees." |
| **RL** | Reinforcement Learning | Training a policy by reward/trial-and-error in simulation. Krabby learns parkour locomotion this way. |
| **RSI** | Reference State Initialization | Starting a fraction of episode resets (`KRABBY_RSI_FRAC`) from a bank of states sampled along a scripted reference gait, so the policy starts inside the gait instead of from a stand. |
| **S3** | (AWS) Simple Storage Service | Object storage; the bench smoke test compares deployed firmware against a manifest kept in S3. |
| **SDK** | Software Development Kit | The Python library you call from code (e.g. `KrabbyMCUSDK`). The CLI is a thin wrapper around it. |
| **Shield** | — | A board that plugs on top of an Arduino to add hardware. The Krabby-Uno shield carries motor-driver and sensor wiring; it has no CPU of its own. |
| **Site geometry (rough terrain)** | — | `KRABBY_FLAT_TERRAIN_GEOM=site`, the third omni terrain geometry beside `ring` and `field` (`mdp/omni_terrain_knobs.py`; generator `parkour_isaaclab/terrains/omni/site_terrains.py`; default off). With `KRABBY_SITE_LADDER=rt` and a `KRABBY_SITE_MIX` it builds the rough-terrain course c1 and C1S trained on: seven hazard kinds on 10-level ladders (slopes 2 to 20 deg, hills, steps up 0.02 to 0.30 m, steps down 0.02 to 0.35 m, potholes to 0.35 m deep, gravel, rubble cells 0.02 to 0.15 m) beside flat and legacy-field columns, no walls or blockers. The training mix is 20 columns: 3 flat, 3 field, 2 of each hazard kind. Kind rows evaluate one kind at a time (`KRABBY_SITE_MIX=<kind>=20`) and need this branch's code. |
| **SSM** | (AWS) Systems Manager — Parameter Store | Where the bench fetches its credentials/config from, instead of baking secrets into the device. |
| **Stand network** | — | A second omni policy routed on stand holds beside the walking head, so the robot stands still and homed (stand network of record: `policy/rt_stand_net/`, SH_RT, warm-started from SN_RT, both re-paired beside c1). |
| **T\*** | — | The walking head selected by the Task 4 pipeline's payload rule (F19's walker of record, 2026-10-24 to 2026-11-09): `f19_8_p3b_payload180/model_25992.pt` (seed 19). It is the resume point of TDR. |
| **TDR** | — | The omni walker of record from 2026-11-09 to 2026-11-14, and the resume point of c1: T\* fine-tuned for 3000 iterations with per-env rod-speed domain randomization (rod slew x U(0.6, 1.35), `KRABBY_DR_ROD_SPEED`, training-only): `rsd_9_p3b_rodspeed_dr/model_28991.pt` (seed 19; a live stage). |
| **TensorRT** | — | NVIDIA's inference optimizer that runs the trained policy fast on the Orin (typically from an ONNX export). |
| **ToF** | Time-of-Flight | A depth-sensing method (used by the MaixSense-A075V camera) that measures distance from how long light takes to bounce back. |
| **ZED 2i** | — | The Stereolabs stereo camera that provides the robot's front RGB-D feed and IMU data. |
| **ZMQ** | ZeroMQ | The lightweight messaging library the HAL uses to pass observations and joint commands between processes. |
