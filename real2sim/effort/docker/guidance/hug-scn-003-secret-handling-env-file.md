---
xid: HUG-SCN-003
kind: hug
effort: scn
status: active
date: 2026-06-03
author: krabby handoff 2026-04-29
bd-id: krabby-ijl
title: Secrets via runtime --env-file, never ENV literal
---

# Secrets via runtime --env-file, never ENV literal in Dockerfiles

## Context
A HuggingFace token was hardcoded as `ENV HF_TOKEN=...` in four Dockerfiles (`Dockerfile`, `.mast3r`, `.slam3r`, `.vggt`). The files were never committed (status `??`), so the token never reached a remote; the user opted not to rotate. Resolved 2026-04-29.

## Direction
- NEVER use `ENV <secret>` literal, and avoid build-args for secrets (they land in image metadata too).
- Keep secrets in a gitignored `.env` at the repo root with a committed `.env.example` template; `.gitignore` carries `.env` / `.env.*` with `!.env.example`.
- Consume at runtime: `docker run --env-file ../.env --gpus all --shm-size=8g <image>`.

_Source: krabby/archive/rotate-hf-token-slam3r-dockerfile.md._
