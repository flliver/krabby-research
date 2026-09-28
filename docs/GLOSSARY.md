# Krabby Glossary

A quick-reference glossary of acronyms and terms used across the Krabby project.

For in-depth explanations of the stack see [TECHNOLOGY_AND_TERMINOLOGY.md](TECHNOLOGY_AND_TERMINOLOGY.md).

---

| Term | Expansion | What it means in Krabby |
|------|-----------|-------------------------|
| **ADC** | Analog-to-Digital Converter | The MCU circuit that turns an analog voltage (a pot wiper, an IS line) into a number 0–1023. This is what `analogRead()` returns. |
| **CAL** | Calibration | Finding each joint's real min/max sensor values so a normalized command (0.0–1.0) maps onto its actual travel. |
| **CLI** | Command-Line Interface | The terminal command wrapper around the SDK (e.g. `python -m firmware …`). |
| **CRC** | Cyclic Redundancy Check | A checksum stored alongside EEPROM data to detect corruption; a bad CRC means "don't trust this saved calibration." |
| **DoF** | Degrees of Freedom | The number of independently controllable joints. Each Krabby leg has three (yaw, hip, knee). |
| **DR** | Domain Randomization | Randomized pushes, chassis mass and CoM offsets during training (`KRABBY_DR_PUSH` / `KRABBY_DR_MASS` / `KRABBY_DR_COM`) so the policy tolerates model error. |
| **ECR** | (AWS) Elastic Container Registry | Where built locomotion container images are published. The bench watchdog polls it for new images to deploy. |
| **EEPROM** | Electrically Erasable Programmable Read-Only Memory | Tiny non-volatile memory on the MCU. Krabby stores each board's role/serial and per-joint calibration here so it survives power-off. |
| **EMI** | Electromagnetic Interference | Electrical noise (e.g. from a running motor) that a floating/unconnected sensor pin can pick up — a source of false readings the firmware has to guard against. |
| **Exposure (obstacle)** | — | Training-time telemetry of how often episodes leave the start platform and reach each obstacle (`Metrics/<term>/reach_edge_frac`, `reach_obst_frac`, `obst_coverage_k`). |
| **Gait clock** | — | A per-env phase advanced each step at a cadence linear in the commanded forward speed (frozen on stop commands); the policy observes its sin/cos and the clock rewards pay for a tripod contact schedule locked to it. |
| **HAL** | Hardware Abstraction Layer | The boundary that lets the *same* policy run against either simulation or real hardware. Talks over ZMQ; swap the backend, not the policy. |
| **Hall (sensor)** | Hall-effect sensor | A magnetic position sensor. On Krabby it gives *incremental* counts (relative motion), used on the hip-lift and yaw joints. |
| **H-bridge** | — | The motor-driver circuit (one per motor) that lets the MCU drive a motor forward or backward. Also outputs the current-sense ("IS") signal. |
| **Hexapod / "hex"** | — | A six-legged robot. Krabby's target chassis. The **training plant** is `assets/crab.usda` (generated, A15+B geometry; see [crab-hexapod-plant.md](crab-hexapod-plant.md)); `assets/crab_hex_ref.*` is the legacy teleop/HAL demo model. |
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
| **SSM** | (AWS) Systems Manager — Parameter Store | Where the bench fetches its credentials/config from, instead of baking secrets into the device. |
| **TensorRT** | — | NVIDIA's inference optimizer that runs the trained policy fast on the Orin (typically from an ONNX export). |
| **ToF** | Time-of-Flight | A depth-sensing method (used by the MaixSense-A075V camera) that measures distance from how long light takes to bounce back. |
| **ZED 2i** | — | The Stereolabs stereo camera that provides the robot's front RGB-D feed and IMU data. |
| **ZMQ** | ZeroMQ | The lightweight messaging library the HAL uses to pass observations and joint commands between processes. |
