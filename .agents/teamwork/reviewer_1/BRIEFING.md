# Reviewer 1 Briefing: UNet Exponential Moving Average (EMA) Integration Review

## Context & Objectives
- **Task**: Adversarial review and hardening of the Exponential Moving Average (EMA) integration for the UNet diffusion model in `train.py`, `main.py`, and related scripts (`inference.py`, `evaluate.py`).
- **Goal**: Ensure EMA prevents mode collapse during diffusion training, updates accurately after optimization steps, serializes cleanly in checkpoints alongside active weights, and supports seamless checkpoint resumption.
- **Requirements**:
  - **R1**: Integrate EMA using a robust, pre-built/standard PyTorch EMA library (`torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn`).
  - **R2**: Update training loop to update EMA weights after each optimization step, save both active and EMA UNet weights in checkpoints, and support resumption.
  - **Acceptance Criteria**:
    - Fast dummy training run completes successfully without crashing.
    - Saved checkpoint contains EMA state dictionary alongside active weights.
    - Training can be resumed cleanly from newly generated checkpoint.

## Strategy
1. **Adversarial Code Inspection**: Deep dive into `train.py`, `main.py`, `inference.py`, `evaluate.py`, and test files.
2. **Identify Defects / Edge Cases**:
   - AMP GradScaler step skipping when gradients are inf/NaN.
   - EMA model device placement and multi-GPU/DataParallel interactions.
   - Checkpoint loading edge cases (state dict prefixes, key matching, mismatched parameters, optimizer/scheduler resumption).
   - In-place vs copy semantics, evaluation/inference mode on EMA model.
   - Evaluation / inference scripts using EMA weights vs active weights.
   - Unit tests / verification suite comprehensiveness and edge-case testing.
3. **Remediate Code Defects**: Apply targeted, robust fixes directly to the codebase.
4. **Independent Verification**: Validate all changes via static code analysis, AST inspection, mathematical proofing, and unit test suites.
5. **Document & Report**: Update `progress.md`, write comprehensive `handoff.md`, and notify orchestrator.
