# Sherpa Knowledge Base

Persistent knowledge store for krabby's **sherpa** (🏔️) — the project's
topology guide. Index entries here as they accumulate.

## Index

- [`consumption-map.md`](consumption-map.md) — what krabby **binds to**
  from the fleet's `shared` exposures (baeprz hosts s/t/b/d/j, mqtt,
  nanny-progress), discovered via `ccc-bd topology` rather than
  hardcoded. The "bind by discovery, not by hostname" stance. krabby is
  primarily a **consumer**, so this is the load-bearing doc.

## Operating reminders

- **Live truth is the CLI.** `ccc-bd topology ls` is always
  authoritative; knowledge files name *which* exposures matter and
  *why*, never a frozen copy. Re-discover when in doubt.
- **Point, don't broker.** Answer who-exposes-what + name the
  `contact` (`agent@project`); route via `/notify <project>`. Never call
  another project's service on someone's behalf.
- **Never speak a secret** (HUG-SHP-004). Share `endpoint` hints + the
  `access_script` *path* only — never read, echo, or infer credentials.
- krabby's own exposures live in [`../../../topology.json`](../../../topology.json).
