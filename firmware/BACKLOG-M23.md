# Milestone 23 — Combined improvement backlog

**Status:** Ordered and packed. Work top-down from the cut line.

**Sources**


| Source        | Path                                                                                                                   | Role here                                   |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Friction log  | `[FRICTION-LOG.md](FRICTION-LOG.md)`                                                                                   | Task 1 cold-start / bench deviations (`F*`) |
| Known issues  | `[KNOWN-ISSUES.md](../../patina-foundation-grants/grants/Krabby-Uno/Milestone23-FleetReliability-QOL/KNOWN-ISSUES.md)` | Pre-M23 inventory (`K*`)                    |
| M21 hand-offs | Overview “Looking ahead”                                                                                               | Scored as candidates, same rubric (`M21-*`) |


**Scoring rubric** 


| Field    | Meaning                                                                                      |
| -------- | -------------------------------------------------------------------------------------------- |
| Blocking | Stops bring-up / release / CI vs only slows it                                               |
| Cost     | Minutes or hours lost, and how often it recurs                                               |
| Effort   | Hours to fix properly (whole hours only; round fractions **up**) — estimate used for packing |
| Actual   | Hours really spent (incl. returned items); fill when the item closes                         |
| Who      | AI-suited / Human / Hardware                                                                 |
| Priority | install/execution blockers → intermittent → bad warnings → documentation. Impact over effort |


---



## 60-hour cut line

The **60 hours** are measured by **actual hours spent**, not by how many estimated rows close. Effort estimates pack the starting cut only. As Actual (h) is logged, items may be **pulled up** from below the line if hours remain, or **pushed down** / returned if actuals overrun — revisit the cut as the bucket is worked.

Packed total above the line: **66.0 h** estimated (within 10% of 60).




---



## Ordered list (work top-down)

Status: `open` · `in progress` · `done` · `returned` · `out of scope`. `Cumul` is estimated effort only.


| Pri | ID | Source | Summary | Blocking | Cost (with recurrence) | Effort (h) | Actual (h) | Who | Cumul (h) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | F4 | friction | Leader `J` forward / follower parse: LEFT/RIGHT H-bridges silent under GUI jog | was yes for GUI jog | ~1–2 h misdiagnosing UART; every 3-board GUI session on unpatched FW | 4 | 4 | Hardware + Human diagnose; AI assisted fix; Human reflash | 4.0 | done |
| 2 | F1 | friction | README `pip install` needs venv on Orin | yes | ~5 min; every fresh Orin | 2 | 0.5 | AI docs; Human | 6.0 | done |
| 3 | F2 | friction | `sudo krabby install` hits wrong/`No such command` binary | yes | ~10 min; every venv + sudo | 1 | 0.5 | AI docs; Human | 7.0 | done |
| 4 | K28 | known | No top-level “read this for your intent” routing | yes (findability) | every new reader; cheapest high-impact doc fix | 1 | 0.5 | AI; Human | 8.0 | done |
| 5 | K29 | known | Spoken `pip install krabby` ≠ `krabby-launcher` | yes (trap) | every new user who guesses the spoken name | 2 | — | AI docs loud; Human | 10.0 | open |
| 6 | F3 | friction | GUI + Pro pair script not on PyPI; kit path needs clone | yes without clone (GUI/pair) | ~15–30 min; every kit-only bring-up | 2 | 3 | AI package/docs; Human PyPI/kit | 12.0 | done |
| 7 | F6 | friction | Boot unit owns `/krabby`; manual `krabby run` conflicts | yes (fresh manual run) | ~2–5 min; every boot + manual run | 2 | — | AI docs/code; Human | 14.0 | open |
| 8 | F5 | friction | Pro Controller `Paired: no` / Sync vs cache; warn blames `hid_nintendo` | yes (gamepad drive) | ~15–30 min; every first BT pair / bad reconnect | 3 | — | AI warn/docs; Human+HW | 17.0 | open |
| 9 | F13 | friction | Pro Controller idle power-off → Home wake; HAL / `krabby run` does not reopen joystick | yes for drive after idle sleep | ~1–3 min restart; every Pro idle power-off mid-session | 16 | — | AI reopen/hotplug; Human+HW | 33.0 | open |
| 10 | F12 | friction | Artifact-health **locomotion-image** stage FAIL | yes (green artifact-health) | ~15–60 min once Actions log read | 4 | — | AI workflow; Human Discord PASS | 37.0 | open |
| 11 | F11 | friction | `docker/setup-qemu-action@v3` Node 20 deprecation warn | no until Node 20 removal | ~5 min when editing workflow | 4 | — | AI bump; Human CI green | 41.0 | open |
| 12 | K1 | known | Default branch `main` publishes nothing; workflows watch `mainline`/`release/**` | yes (silent no-ship) | every merge to `main`; hours of false confidence | 6 | — | Human decision + AI change/docs | 47.0 | open |
| 13 | K5 | known | Isaac `contacvt_sensor` typo; contact never assigned | yes (sim contact path) | every contact/current read through Isaac HAL | 1 | — | AI; Human | 48.0 | open |
| 14 | K13 | known | Firmware error branch silently swallowed (`arduino.ino`) | yes when hit | can burn a bring-up afternoon | 6 | — | AI + Hardware; Human | 54.0 | open |
| 15 | K17 | known | Gamepad mapper assumes start joints; does not read state | no | wrong start-pose assumptions | 3 | — | AI + Hardware; Human | 57.0 | open |
| 16 | F9 | friction | Mid-upload interrupt / hub enum: board “vanishes” (docs-only fix) | yes until reappears | ~15–45 min per stuck board | 2 | — | AI docs (App. B); Human | 59.0 | open |
| 17 | F7 | friction | `KRABBY_MCU_PORT` not forwarded by `gamepad_cmd` / `krabby run` | no if auto-detect works; yes if operator trusts warn | ~5–15 min; every CH340/`ttyUSB*` | 4 | — | AI forward env; Human+HW | 63.0 | open |
| 18 | F8 | friction | `XDG_RUNTIME_DIR` error on `krabby run` (cosmetic) | no | ~0–1 min; every `krabby run` | 3 | — | AI container env; Human | **66.0** | open |

### ——— cut line ———


| Pri | ID | Source | Summary | Blocking | Cost (with recurrence) | Effort (h) | Actual (h) | Who | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | K12 | known | Actuators instantiated before role election finishes | intermittent | wrong first wiring until corrected | 16 | — | Hardware + AI; Human | open |
| 20 | K3 | known | Publish workflows still silent (bench Discord ≠ publish notify) | no | Actions tab or try-to-use only signal | 5 | — | AI; Human | open |
| 21 | K2 | known | No test suite on push/PR (only on package version tags) | no (until release) | every breaking commit until someone tags | 5 | — | AI + Human | open |
| 22 | K27 | known | Root README interleaves user / developer / infra | yes (cold-start clarity) | every new user; residual after F1/F2 line fixes | 3 | — | AI + Human | open |
| 23 | K33 | known | Two competing host-setup paths (manual Jetson doc vs `krabby install`) | yes (wrong path) | hours if reader follows the stale guide | 2 | — | Human decide + AI mark/doc | open |
| 24 | K31 | known | Enrollment documented in `ENROLL.md` and `SETUP-FLEET.md` | yes (which is current?) | drift; Task 1 had to pick one | 2 | — | AI; Human | open |
| 25 | K30 | known | `firmware/SETUP.md` milestone title + four audiences | no | confusion every firmware reader | 3 | — | AI; Human | open |
| 26 | K34 | known | `DEVELOPER.md` untitled; sim-only half of development | no | real-robot dev path undocumented | 3 | — | AI; Human | open |
| 27 | K32 | known | `SETUP-FLEET.md` mixes account setup and per-robot ops | no | infra vs operator confusion | 3 | — | AI; Human | open |
| 28 | K35 | known | Image/package READMEs mix consumer and maintainer | no | wrong audience sections | 2 | — | AI; Human | open |
| 29 | K26 | known | M21 contract says merge to `main` (follow K1 decision) | tied to K1 | contract/docs wrong until aligned | 1 | — | AI (after K1); Human | open |
| 30 | K19 | known | Gamepad launch hard-codes Jetson HAL backend | yes (non-Jetson gamepad) | cannot launch other backends via that path | 16 | — | AI; Human | open |
| 31 | K16 | known | Observation timestamps not propagated (`None`) | no | incomplete teleop latency accounting | 2 | — | AI; Human | open |
| 32 | K18 | known | Gamepad path tested on macOS only (Orin partly covered in T1) | partial | Linux/Windows still thin | 4 | — | Hardware; Human | open |
| 33 | M21-B | M21 | Absent MCU → detect and offer simulator | yes for no-MCU machines | cannot soft-fail to sim today | 16 | — | AI + Human | open |
| 34 | K21 / M21-D | known / M21 | Stopped simulator stays offline; no IoT resume | yes for sim recovery | manual CLI restart each time | 16 | — | AI + Human | open |
| 35 | K10 | known | John’s OLED simulator referenced, not in repo | yes for M21 T4 | blocks until located/vendored | 6 | — | Human | open |
| 36 | K4 / M21-A | known / M21 | IsaacSim image built locally, never published | no for hardware path | every sim adopter rebuilds (hours) | 10 | — | AI + Human | open |
| 37 | M21-C | M21 | GPU capability validation before launch | no | failed launches on wrong GPU | 6 | — | AI + Human | open |
| 38 | K6 | known | Real MCU position commands disabled; jog fallback | architectural | closed-loop path not what robot runs | 16 | — | Human + Hardware | open |
| 39 | K11 | known | Actuator identity / pin config hardcoded in several places | no | hours whenever roles/names change | 24 | — | AI + Hardware; Human | open |
| 40 | K8 | known | HAL has no abstract backend contract | no | days of reading to add a backend | 24 | — | AI + Human | open |
| 41 | K7 | known | Jetson observations partly unimplemented | no | incomplete real sensor obs | 32 | — | Human + Hardware | open |
| 42 | K15 | known | `MODEL_CONTROLLER_KRABBY` unimplemented stub | feature | blocks model-controller mode | 24 | — | Human | open |
| 43 | K9 | known | M16 sensors specified, not in `firmware/arduino/` | yes for M21 T4 sim | large cross-milestone gap | 40 | — | Human + Hardware | open |
| 44 | K20 | known | Updates require SSH (`krabby update`); no cloud rollout | scope decision | every update needs an operator | 20 | — | Human | open |
| 45 | K29-PEP541 | known | Request PyPI `krabby` name under PEP 541 | no (docs fix is above) | process time; discretionary transfer | 4 | — | Human | open |
| 46 | K14 | known | Actuator lookup linear/quadratic (OK at six) | no | none until actuator count grows | 1 | — | AI; Human | open |
| 47 | K24 | known | Grants README describes `T1-` / `AUDIT-LOG` nobody uses | no (meta) | contributor naming mismatch | 1 | — | AI; Human | open |
| 48 | K25 | known | Grant time-estimate drift (checked in M21; others unchecked) | no (meta) | estimate inconsistency | 1 | — | Human | open |
| 49 | K22 | known | Windows `play.py` needs three undocumented fixes | Windows only | hours on first Windows setup; no local Windows — need machine/access | 10 | — | Human (needs Windows); AI assist | open |
| 50 | K23 | known | Kit non-fatal sensor extension DLL errors on Windows | Windows only | minutes of false alarm; needs Windows machine to verify docs | 16 | — | Human (needs Windows); AI docs | open |
| 51 | F10 | friction | `firmware show` sometimes only FRONT until USB replug | yes (3-role flash/bring-up) | ~1–2 min; intermittent | 3 | — | Hardware + Human; AI docs/retry | in progress — do not move to Tier 1 (someone else working on it); close when finished |

## Out of scope / done (explicit)


| ID                     | Reason                                                                                                    |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| Task 1 `1d`            | Self-hosted Orin runner — `FRICTION-LOG.md` Task 1 open items / Appendix C; not this Task 2 bucket        |
| Task 1 `1a`            | Front teleop video (ZED USB 3) — `FRICTION-LOG.md` Task 1 open items / Appendix D; not this Task 2 bucket |
| K36                    | Done as Task 1 guidance (extend `krabby-bench`); not a remaining fix                                      |
| K20 (as “reverse M10”) | Out of scope unless reducing SSH friction *without* cloud-driven rollout                                  |


