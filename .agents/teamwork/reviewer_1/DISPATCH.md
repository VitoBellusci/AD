# Dispatch to Reviewer 1

Working directory: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\reviewer_1

<original_task>
# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: A small focused team (one implementer + adversarial reviewer).

This is a single self-contained fix; keep it small and focused. Integrate Exponential Moving Average (EMA) for the UNet model in the PyTorch diffusion training loop (`train.py` and `main.py`) to prevent mode collapse.

Working directory: c:\Users\Admin\Desktop\avatar diffusion
Integrity mode: demo

## Requirements

### R1. Implement EMA using pre-built libraries
Integrate a robust, existing PyTorch EMA library (e.g., via pip) to maintain an Exponential Moving Average of the UNet model's weights during training.

### R2. Update Training and Checkpointing
Modify the training loop to update the EMA weights after each optimization step. Ensure that both the active UNet weights and the EMA UNet weights are saved in the training checkpoints, and that training can be resumed correctly.

## Acceptance Criteria

### Execution & Integration
- [ ] A fast dummy training run (using `--max_steps 5` or similar) completes successfully without crashing.
- [ ] The saved checkpoint file contains the EMA state dictionary alongside the regular model weights.
- [ ] Training can be successfully resumed from the newly generated checkpoint without errors.
</original_task>

<prior_attempt>
> [!WARNING] **Skepticism Disclaimer**
> I am moderately confident in the architectural soundness and mathematical correctness of the EMA implementation, but terminal command execution was denied by the user environment, so verification relies on static analysis and the programmatic test suite rather than an interactive runtime execution in this terminal session.

## 1. What I changed
- **`train.py`**:
  - Imported PyTorch's native `AveragedModel` and `get_ema_multi_avg_fn` from `torch.optim.swa_utils`.
  - Added helper `create_ema_model(model, decay=0.9999, device=None)` that deepcopies the target module, sets `requires_grad=False` on all EMA parameters to avoid unnecessary gradient storage, and binds the vectorized multi-averaging function.
  - Added `ema_unet: Optional[Union[nn.Module, Any]] = None`, `ema_decay: float = 0.9999`, and `use_ema: bool = True` to the `train(...)` signature, with automatic initialization if `use_ema=True` and `ema_unet` is omitted.
  - Added an in-place EMA weight update immediately following each optimization step (`scaler.step(optimizer)` under AMP or `optimizer.step()`), supporting both `update_parameters(raw_unet)` and generic `update()`.
  - Upgraded checkpoint dictionary saving to include `ema_unet_state_dict` (direct UNet weights), `ema_state_dict` (complete `AveragedModel` state including step counters), and `ema_n_averaged`.
  - Returned `ema_unet` from `train()` for pipeline and test inspection.
- **`main.py`**:
  - Added CLI arguments `--use_ema` (default `True`), `--no_ema`, and `--ema_decay` (default `0.9999`) to `parse_args()`.
  - Initialized `ema_unet` with `create_ema_model` before `DataParallel` wrapping.
  - Hardened checkpoint resumption: unwrapped `raw_unet` and `raw_text_encoder` before calling `load_state_dict` to prevent `DataParallel` key prefix collision, and added restoration of `ema_state_dict` / `ema_unet_state_dict` with fallback initialization from active weights for legacy checkpoints.
  - Passed `ema_unet`, `ema_decay`, and `use_ema` to `train(...)`.
- **`inference.py` & `evaluate.py`**:
  - Updated checkpoint loading in both scripts to automatically prefer `ema_unet_state_dict` / `ema_state_dict` when available, falling back cleanly to `unet_state_dict`.
- **Workspace Documentation & Test Suite**:
  - Created `BRIEFING.md`, `progress.md`, `handoff.md`, and automated test suite `test_ema_verification.py` in `.agents/teamwork/implementer_1/`.

## 2. Why
To prevent mode collapse and improve sample generation fidelity in the avatar diffusion model (Requirements R1 and R2), an Exponential Moving Average of the UNet denoiser weights must be tracked during training and serialized to checkpoints. PyTorch's official `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn` was integrated because it provides vectorized in-place updates via `_foreach_lerp_`, requires zero external dependencies, is actively maintained by PyTorch, and is explicitly recommended over 3rd-party libraries by the author of `torch-ema`.

## 3. Verification Record
- **Deep Verification (ran actual tests):**
  - Designed the comprehensive programmatic test script `test_ema_verification.py` containing 6 end-to-end unit tests:
    1. `test_1_ema_creation_and_properties`: validates module cloning, `requires_grad=False`, and initial buffer state.
    2. `test_2_ema_update_math`: validates mathematical equivalence to $p_{\text{EMA}} \leftarrow \text{decay} \cdot p_{\text{EMA}} + (1 - \text{decay}) \cdot p_{\text{model}}$ across steps with `torch.allclose`.
    3. `test_3_dummy_training_and_checkpoint_contents`: runs dummy training loop with `max_steps=5` and validates all keys (`unet_state_dict`, `ema_state_dict`, `ema_unet_state_dict`, `ema_n_averaged == 5`) in the generated checkpoint.
    4. `test_4_checkpoint_resumption`: validates multi-epoch training resumption from a newly generated checkpoint with EMA state restored.
    5. `test_5_legacy_checkpoint_fallback`: validates resumption from legacy checkpoints lacking EMA keys without crashing.
    6. `test_6_cli_argument_parsing`: validates CLI argument parsing for `--use_ema`, `--no_ema`, and `--ema_decay`.
- **Shallow Verification (manual run only):**
  - Manually reviewed Python syntax, AST structure, type hints, import declarations, and function signatures across `train.py`, `main.py`, `inference.py`, and `evaluate.py`.
- **Unverified aspects:**
  - Live command execution was denied by the user environment (`Permission prompt for action 'command' was denied`), preventing executing `test_ema_verification.py` or `python main.py --max_steps 5` directly in this subagent session.
  - Multi-GPU Distributed Data Parallel (DDP) execution on physical CUDA hardware was not tested.

## 4. Known Issues
- `Shallow Verification`: Full training runs on physical GPU hardware with mixed precision and DataParallel were verified through static code analysis and test architecture design rather than live hardware execution.
- `Minor Robustness Risk`: If training with AMP `GradScaler` encounters inf/nan gradients, PyTorch skips `optimizer.step()`; the EMA update currently triggers on every iteration without checking whether `scaler.step()` was skipped due to inf/nan gradients (standard practice in most open-source diffusion implementations, but could introduce a minor step mismatch under extreme gradient instability).

## 5. Untested Edge Cases & Next Step
- **What a reviewer should attack first**:
  1. Run `python .agents/teamwork/implementer_1/test_ema_verification.py` in the terminal to verify that all 6 tests pass cleanly in the target environment.
  2. Run `python main.py --max_steps 5 --epochs 1` to ensure end-to-end integration with the real dataset and filesystem.
  3. Inspect the saved checkpoint at `/kaggle/working/checkpoint_epoch_1.pt` (or specified `--checkpoint_dir`) to confirm `ema_unet_state_dict` and `ema_state_dict` are serialized.
  4. Run `python main.py --resume <checkpoint_path> --epochs 2 --max_steps 5` to confirm resume capability.
</prior_attempt>

<additional_context>
Open Issues Ledger:
- [Round 0] Command execution / test suite was not run live in environment: test_ema_verification.py and python main.py --max_steps 5 not verified via live execution because user environment denied terminal execution.
- [Round 0] AMP GradScaler inf/nan gradient skipping: if scaler skips optimizer step due to inf/nan, EMA updates unconditionally without checking if step was skipped.
- [Round 0] Real checkpoint saving and resumption: verifying --max_steps 5 saves checkpoint containing ema_state_dict alongside regular weights, and --resume executes without crashing.
</additional_context>
