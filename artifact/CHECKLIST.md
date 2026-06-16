# Artifact Checklist

## Packaging

- [x] The artifact has a single entry point: `artifact/README.md`.
- [x] The README starts with a reviewer fast path.
- [x] The README includes compact installation commands.
- [x] Long setup and troubleshooting notes are moved to `artifact/ARTIFACT_APPENDIX.md`.
- [x] The README includes hardware and proprietary dependency requirements.
- [x] The README explains BM1690 PCIe performance mode.
- [x] The README explains cmodel correctness-only mode.
- [x] Paper figures are copied into `artifact/figures/paper/`.
- [x] Generated results are ignored under `artifact/results/`.

## Functional Checks

- [x] Environment checker exists: `artifact/scripts/check_env.py`.
- [x] Smoke runner exists: `artifact/scripts/run_smoke.py`.
- [x] Core runner exists: `artifact/scripts/run_core.py`.
- [x] Ablation runner exists: `artifact/scripts/run_ablation.py`.
- [x] Result parser exists: `artifact/scripts/parse_results.py`.
- [x] Manifest verifier exists: `artifact/scripts/verify_manifest.py`.

## Paper Mapping

- [x] The README maps each covered paper figure to a command, expected claim, and copied figure file.
- [x] Figure 4 config exists: `artifact/configs/fig4_standalone_ops.json`.
- [x] Figure 5 config exists: `artifact/configs/fig5_fused_ops.json`.
- [x] Figure 6 config exists: `artifact/configs/fig6_ablation.json`.
- [x] Expected claims are recorded in `artifact/expected/paper_claims.json`.
- [x] Correctness tolerances are recorded in `artifact/expected/tolerances.yaml`.

## Review Boundaries

- [x] Cmodel is documented as correctness-only.
- [x] FlashAttention fp32 is excluded from valid performance claims.
- [x] Matmul fp32 semantics are documented.
- [x] Performance variation is documented.
- [x] The artifact avoids card-management and machine-local recovery instructions.
