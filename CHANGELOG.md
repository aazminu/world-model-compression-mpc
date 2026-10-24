# Changelog — `world-model-compression-mpc`

All notable architectural, verification, and release changes to `world-model-compression-mpc` are recorded here.

## [0.3.0] — 2026-10-24
### Added
- Added longitudinal release changelog (`CHANGELOG.md`) and expanded CLI reproducibility & verification documentation.

## [0.2.1] — 2026-10-22
### Added
- Expanded automated `pytest` regression coverage for deterministic PRNG seed propagation, CLI `--format json` schema invariants, and `--output` I/O error handling.

## [0.2.0] — 2026-10-19
### Changed
- Wired deterministic PRNG seed controls into the benchmark harness and hardened CLI `--output` file creation and exception reporting.

## [0.1.0] — 2026-10-15
### Added
- Initial architecture and benchmark suite: Activation-Aware SVD (ASVD) latent world-model compression and Hybrid Forward-KL + Elite-Classification distillation for CEM MPC.
