# Third-party components and provenance

This firmware subproject contains multiple licenses. The surrounding repository's LICENSE is not a replacement for component-specific notices. This inventory does not relicense any source or provide legal clearance for a commercial release.

| Component | Location | Existing notice / provenance |
|---|---|---|
| OpenBLT target and inspection references | vendor/openblt | Pinned revision in UPSTREAM.txt; retain Doc/license.html and file headers, including the upstream modified GPL text/exception |
| Krabby OpenBLT integration | Bootloader | GPL-3.0-or-later header in main.c; modifications/local files are here, not edits to the selected vendor target sources |
| CMSIS | Drivers/CMSIS | Retained LICENSE.txt and device-pack license; retain per-file notices |
| STM32G4 HAL | Drivers/STM32G4xx_HAL_Driver | Retained LICENSE.txt and per-file notices |
| STM32 USB middleware | Middlewares/ST/STM32_USB_Device_Library | Source headers cite ST SLA0044: https://www.st.com/SLA0044 ; this selected copy lacks a standalone license document |
| CubeMX-generated project files | Src, Inc, startup, cmake | Retain ST notices and confirm package-specific terms when preparing distribution |

OpenBLT upstream: https://github.com/feaser/openblt/tree/8fc277c74298a2fedeb99e85f906d846bf0018e3 . Official host-library build instructions: https://www.feaser.com/openblt/doku.php?id=manual:libopenblt . The selected host reference files are not a complete host build source tree.

The local upstream libopenblt.dll and BootCommander.exe are excluded from Git. Do not force-add them as part of a source-only commit: exact binary provenance/complete corresponding-source packaging has not been established here. Obtain/build the runtime separately as described in README.md. If distributing those binaries later, review their applicable license and provide the appropriate source/license package first.

Before a public distribution, restore the STM32 USB middleware's exact package license document and confirm its originating package/version. The .ioc records STM32Cube FW_G4 V1.6.3 (CubeMX 6.18.1), but this does not establish the provenance of every selected middleware file. The file headers retain the license link, but this audit did not obtain or certify that full package. Also confirm how the surrounding repository's license is intended to cover original project files. No broad project LICENSE file was invented by the audit.

The official SLA0044 link resolves to https://www.st.com/resource/en/license_agreement/SLA0044.txt (Rev5/February 2018). The audit could read it through web research, but direct download to this workspace failed; no incomplete or reconstructed license text was substituted.
