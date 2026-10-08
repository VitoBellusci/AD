# Implementer 1 Handoff Report: v-prediction (Velocity Target) Implementation

## 1. Executive Summary
Modified the Avatar Diffusion model from noise ($\epsilon$) prediction to velocity ($v$) prediction (v-parameterization, Salimans & Ho 2022).
All three requirements (R1, R2, R3) and acceptance criteria have been implemented, verified mathematically, and tested through execution.

## 2. Requirements Traceability

### R1. Update Training Target (`train.py`)
- Formulated velocity target:
  $$v_{\text{target}} = \sqrt{\bar{\alpha}_t} \cdot \text{noise} - \sqrt{1 - \bar{\alpha}_t} \cdot \text{original}$$
  where `original` is `images` in the training loop and `val_images` in the validation loop.
- Updated both the training optimization loss and the validation tracking loss in `train.py` from `criterion(predicted_noise, noise)` to `criterion(predicted_v, v_target)`.
- Added `if __name__ == "__main__":` entrypoint to `train.py` delegating to `main.py` (with automatic `epochs=1` default when `--max_steps` is passed without explicit `--epochs`) so that running `python train.py --max_steps 2` executes directly and completes cleanly.

### R2. Update Reverse Process (`models/diffusion.py`)
- Updated `DiffusionReverseProcess.sample` to treat model outputs under Classifier-Free Guidance (CFG) and unconditional passes as velocity predictions $v$:
  $$v_{\text{pred}} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$
- Reconstructed both $\hat{x}_0$ and noise $\hat{\epsilon}$ from predicted velocity $v$ and current state $x_t$:
  $$\hat{x}_0 = \sqrt{\bar{\alpha}_t} x_t - \sqrt{1 - \bar{\alpha}_t} v$$
  $$\hat{\epsilon} = \sqrt{\bar{\alpha}_t} v + \sqrt{1 - \bar{\alpha}_t} x_t$$
- Preserved dynamic range clipping `torch.clamp(pred_x0, -1.0, 1.0)` and posterior mean computation `mean = coef1 * pred_x0 + coef2 * x`.
- Added helper functions `get_velocity` to `DiffusionForwardProcess`, and `predict_x0_from_v` / `predict_noise_from_v` to `DiffusionReverseProcess`.

### R3. Update DDIM Sampling (`evaluate.py` and `inference.py`)
- In `inference.py` (`generate`):
  Updated the DDIM sampling loop to parse model output as velocity $v$, reconstruct both clean estimate $\hat{x}_0 = \sqrt{\bar{\alpha}_t} x_t - \sqrt{1 - \bar{\alpha}_t} v$ and noise $\hat{\epsilon} = \sqrt{\bar{\alpha}_t} v + \sqrt{1 - \bar{\alpha}_t} x_t$, clamp $\hat{x}_0$ to $[-1.0, 1.0]$, and construct the next DDIM latent step:
  $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \hat{\epsilon}$$
- In `evaluate.py` (`sample_batch`):
  Updated the DDIM reverse loop using identical v-parameterization formulas.
- In `evaluate.py` (`load_eval_dataset`):
  Set `use_ram_cache=False` when instantiating `AvatarDataset` to match `main.py` and prevent Windows OS memory mapping resource exhaustion (`error code 1450`).

## 3. Verification & Acceptance Criteria
- [x] **Mathematical Exactness**: Verified in `test_v_prediction.py` across full batch and random timesteps that $(\hat{x}_0, \hat{\epsilon})$ inverted from $v_{\text{target}}$ matches $(x_0, \epsilon)$ with max error $< 5 \times 10^{-7}$.
- [x] **Training Run**: Executed `python train.py --max_steps 2` on CUDA. Completed epoch 1 without shape mismatches or crashes, computing MSE loss and saving checkpoint.
- [x] **Inference Run**: Executed `python inference.py --num_steps 2`. Completed without crashing and generated valid image `outputs/sample_seed42_step2_0.png`.
- [x] **Evaluation Run**: Executed `python evaluate.py --num_samples 2 --batch_size 2 --num_steps 2`. Completed without errors.
