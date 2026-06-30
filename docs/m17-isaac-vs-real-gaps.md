# M17 Task 6 — Isaac-populated vs. empty-on-real observation fields

The parkour model was trained in Isaac Sim, which populates observation fields
the real robot cannot produce. This is the explicit distribution-mismatch list
that M15's domain randomization needs to cover. "Empty on real" means the field
is held at a fixed default in the real HAL path, so the model sees a constant
where sim saw a live (often privileged) signal.

| Field | Source in sim | Real-path status | Default on real |
|-------|---------------|------------------|-----------------|
| `delta_yaw` | Isaac `parkour_event` (target yaw − current yaw) | **None** — no global heading target on real | `0.0` |
| `delta_next_yaw` | Isaac `parkour_event` (next target yaw) | **None** | `0.0` |
| `terrain_type_flag` | Isaac terrain config | **None** — no terrain oracle | `1.0` (treat as non-flat) |
| `flat_terrain_flag` | Isaac terrain config | **None** | `0.0` |
| `scan_features` (132) | Isaac `measured_heights` raycaster | **None until M11 + M15 Task 3** (height-scan from depth) | zeros |
| `privileged_latent` (29) | Isaac (body mass, CoM, friction, motor gains) | **None on real — correctly.** This is exactly what the estimator network is trained to infer; it is never measured on hardware. | n/a (estimated) |
| `contact_forces` (5) | Isaac contact sensor (per-foot normal force) | **Approximated** from MCU current sense — first pass, uncalibrated scale (see audit doc §Contacts) | current-derived |

## Notes

- `privileged_latent` is *intentionally* absent on real — the asymmetric
  actor/estimator design has the estimator fill it from the observable history.
  It is listed here for completeness, not as a gap to close.
- `contact_forces` is the only field M17 actively approximates rather than
  zeroing; see [`m17-hal-model-audit.md`](m17-hal-model-audit.md) for the
  mapping and the placeholder-scale caveat.
- `scan_features` stays zero on real until the depth-derived height scan lands
  (M11 + M15 Task 3). A zero scan reads to the model as "flat ground ahead".
- Everything in this table that defaults to a constant is a randomization target
  for M15: the policy must be robust to these signals being uninformative.
