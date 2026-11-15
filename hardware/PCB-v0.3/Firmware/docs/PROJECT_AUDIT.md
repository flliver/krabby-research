# Pre-commit firmware audit ? 2026-11-12

Scope: hardware/PCB-v0.3/Firmware only. Parent Git root is C:/krabby-research and contains extensive unrelated work. No commits, staging, hardware reads, flashing, motor motion, dependency installation, or option-byte changes were performed. Build outputs are in ignored build/audit-* directories. Test evidence: ../bench/project_audit/results.json.

## Assessment

Firmware source and offline tooling pass the checks below. Several repository hygiene/documentation/test defects were corrected. This is a practical bring-up audit, not a full formal safety/security assessment or license certification. Resolve the third-party licensing/provenance item before treating the repository as a complete public distribution.

## Findings corrected

| Finding | Correction |
|---|---|
| Existing binaries had flash-to-RAM padding | Previous cleanup corrected generation and seven files. Rechecked no .bin exceeds physical flash; current normal/LoadMeFirst binaries remain compact. |
| test_application.py still expected schema 2/capabilities 31 | Updated to current protocol 1/schema 4/capabilities 63. Physical suite not run during this audit. |
| Destructive physical scripts ran immediately with no explicit hardware switch | test_application.py, test_application_stress.py, test_openblt.py now require --hardware. Missing-flag exit verified without accessing hardware. |
| Tests hardcoded a developer-specific CubeProgrammer installation | Shared bench_tools.py uses STM32_PROGRAMMER_CLI, PATH, then current-user Windows Cube bundles. |
| Synthetic GUI tests rewrote bench evidence on every run | They now print offline results; only requested physical runs write their bench records. Hash comparison verified no bench JSON changed during final offline tests. |
| No firmware-root README or Python dependency declaration | Added build/use/test instructions and tools/requirements.txt. Added isolated LoadMeFirst CMake preset; normal presets explicitly disable it. |
| Cache/editor/native Windows runtime could be accidentally added | Firmware-local ignores cover these. DLL/EXE remain on disk but are excluded from Git; install instructions explain the fresh-clone dependency. |
| Parent *.log ignore silently omitted intentional bench transcripts | Added a narrow bench/**/*.log exception, leaving ordinary logs ignored. |
| Platform-dependent line ending treatment | Added .gitattributes for source/addressed records and explicit binary artifacts. |
| CubeMX regeneration script hardcoded the developer checkout | It now references the .ioc relative to Firmware working directory. CubeMX regeneration itself was not executed. |
| No quick integrity check for source-controlled images | Added stdlib-only tools/check_artifacts.py: record checksums/address bounds, paired binary equality, package hashes, combined application equality, OpenBLT signature and oversized binary guard. |
| Missing native uploader dependency produced opaque loader errors | Added an actionable DLL-install error and explicit 64-bit Python check. No Linux backend was invented. |

## Verification

Clean directories configured and built from source using ARM GNU 14.3.1+st.2, CMake 4.4.0, Ninja 1.13.2:

| Target | Flash bytes | Clean binary equals existing binary |
|---|---:|---|
| Normal Debug application | 85,512 | Yes |
| OpenBLT Release | 12,000 | Yes |
| LoadMeFirst MinSizeRel, using its new preset | 35,972 | Yes |

The bootloader clean link emits newlib nosys warnings about unsupported read/write/lseek stubs before garbage collection; it links successfully and produces exactly the existing verified bootloader bytes. It has only 288 bytes of its 12 KiB allocation free, so toolchain changes need size/rebuild checks. No linker-size limit was relaxed.

Passed: read-only GUI smoke test; Control/Jog/Position synthetic suites; compileall; all project JSON syntax; 15 S-record checks; 3 paired bench binaries; combined LoadMeFirst HEX; manifest hashes; signed application equality between provisioning files; no .bin larger than 128 KiB. Firmware documents' Markdown links checked with no missing local targets found. Targeted credential-pattern scan found no credentials in project-authored source/docs; this is not a comprehensive secret-scanner guarantee.

The .ioc FDCAN nominal/data timing fields match main.c's 1M/5M values; runtime TDC stays in user-maintained bridge code. Linker application/configuration boundaries and the OpenBLT checksum-slot assertion remain present. No .ioc, motor semantics, flash allocation or embedded firmware behavior changed during this audit.

## Remaining items and boundaries

1. **Public distribution/license packaging:** see ../THIRD_PARTY.md. Keep GPL/OpenBLT and ST notices distinct from the parent Apache-2.0 license. Selected host references are not the complete DLL source. The local host binaries are now ignored, not silently presented as rebuildable vendored components. Restore the USB middleware's exact full license document and confirm package provenance before public redistribution. Direct download of the official SLA0044 text failed; no guessed license was added.
2. **Fresh-clone uploader setup:** copying/building the official 64-bit libopenblt.dll is a documented external dependency. The rest of the Python diagnostics does not need it. No automated download or Windows DLL rebuild was attempted. The current wrapper remains Windows-specific even though upstream supports Linux.
3. **Hardware-only acceptance:** blank-chip DFU programming of LoadMeFirst, CAN replacement from this exact indicator image, multi-node behavior, physical power-cut recovery, mechanical stopping and sustained thermal/load performance remain unverified. Earlier recorded one-board motor/CAN tests are retained, not rerun or relabelled as audit tests.
4. **Fixture scripts:** legacy destructive suites still assume node 0 and COM7/COM3, overwrite configuration/firmware, and are not general-purpose CI. They are explicit and documented now. Do not run them on a live robot as an offline validation step.
5. **Artifacts:** roughly 50 MB of intentional source/bench files remains, including debug-symbol ELF/map files, firmware versions, serial/UID identifiers, configuration snapshots and deliberately corrupt fixtures. These are useful historical evidence, not all current loadable releases. Individual files are far below GitHub's 100 MB file limit. Do not bulk-flash bench/ or use a historical known_good filename as a current-version selector.
6. **Git boundary:** this firmware subproject was still entirely untracked relative to its parent repository at audit time. Existing unrelated parent modifications were not altered. Review the explicit firmware path rather than running git add at repository root indiscriminately. Local AGENTS/editor state and cached builds are not part of the source handoff.

## Before committing

Run the offline commands in ../README.md, inspect ../THIRD_PARTY.md, and review the firmware-only untracked/staged file list from the actual Git root. Do not force-add ignored native runtimes, caches or builds. If publishing a public release, complete the licensing/provenance item first. Do not claim this offline audit re-proves physical motor safety or the untested provisioning sequence.
