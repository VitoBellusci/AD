# Handoff Report: Training, Evaluation & Usability Investigation

**Agent**: Survey Explorer 3  
**Target Milestone**: Survey Milestone 3  
**Working Directory**: `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_3`  
**Handoff Type**: Hard (Investigation complete)  

---

## 1. Observation

Direct observations from forensic analysis of repository files:

1. **`main.py:53-54` — Unpacking Crash Bug**:
   ```python
   splitter = CompositionalSplitter(config)
   train_idx, val_idx, ood_idx = splitter.split(raw_metadata)
   ```
   Whereas in `preprocessing/splitter.py:47`:
   ```python
   return final_train_indices, val_indices, test_ind_indices, ood_indices
   ```
   `splitter.split()` returns a 4-tuple, but `main.py` unpacks into 3 variables. Attempting to run `python main.py` triggers `ValueError: too many values to unpack (expected 3, got 4)`.

2. **`train.py:103-130` — UnboundLocalError in Unconditional Mode**:
   ```python
   if conditional:
       pad_token_id = tokenizer.vocab.get("<PAD>", 0)
       mask = (text_tokens != pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
       context = text_encoder(text_tokens, mask)
       ...
   else:
       context = get_unconditional_context(
           text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
       )
   ...
   predicted_noise = unet(noisy_images, timesteps, context, mask=mask)
   ```
   `mask` is not defined in the `else` branch, causing `UnboundLocalError: local variable 'mask' referenced before assignment` when `conditional=False`.

3. **`train.py:138-139` — Decoupled Gradient Clipping Present**:
   ```python
   torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
   torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
   ```
   Decoupled gradient clipping per Blueprint 3.1 is already present in `train.py`.

4. **`main.py:98-125` — Inlined Parameter Grouping & Warmup**:
   `main.py` implements AdamW with separate `decay_params` and `no_decay_params` (excluding 1D norms and biases) and a `SequentialLR` chaining 5 epochs of `LinearLR` into `CosineAnnealingLR`. However, the helper function `configure_optimizers()` is not modularized.

5. **`main.py:132` — Filesystem-Fragile Checkpoint Resolution**:
   ```python
   latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
   ```
   Relies on `os.path.getctime` rather than regex integer epoch sorting (DEF-20).

6. **`train.py:78-150` — Absence of Validation Loop & AMP (DEF-10, DEF-14)**:
   `train.py` contains no validation data loader, no validation loss computation, no sample image generation, and no `torch.amp.autocast`.

7. **`inference.py:136-139` — Hardcoded Crash on Launch (DEF-08)**:
   ```python
   generator = AvatarGenerator(
       config_path="./preprocessing/preprocessing_config.json", 
       checkpoint_path="checkpoints/checkpoint_epoch_22.pt"
   )
   ```
   `checkpoints/` is empty (`list_dir` confirmed empty). Calling `_load_checkpoint()` raises `FileNotFoundError`.

8. **`inference.py:143-157` — Blocking Interactive Loop (DEF-12)**:
   Uses `while True:` and `input()`. OOD evaluation at lines 160-164 is unreachable after the infinite loop.

9. **`inference.py:33, 75` — Forced 1,000 Steps & Missing DDIM**:
   `DiffusionReverseProcess(num_time_steps=1000)` executes all 1,000 steps; no `--num_steps` or DDIM sampler.

10. **`metrics.py:65-88` — Range Normalization Applied (DEF-18 / Blueprint 2.2)**:
    `_ensure_zero_one_range(images)` independently scales if `images.min() < 0.0` and clamps to `[0.0, 1.0]`.

11. **`metrics.py:131-139` — `AttributeAlignmentEvaluator` Present (DEF-07 / Blueprint 2.4)**:
    Class is defined, but orphaned (not imported or called by any evaluation script).

12. **`evaluate.py:52` — `AvatarDataset` Signature Mismatch**:
    ```python
    dataset = AvatarDataset(data_dir=data_dir, config=config, tokenizer=tokenizer)
    ```
    `preprocessing/dataset.py:16-22` defines:
    `AvatarDataset(image_paths, metadata, tokenizer, config)`. Calling this raises `TypeError: AvatarDataset.__init__() got an unexpected keyword argument 'data_dir'`.

13. **`models/diffusion.py:75, 143` — Undefined `beta_t` & Missing Mask Signature**:
    Line 75: `sample()` does not accept `mask` or `uncond_mask`. Calling `sample(..., mask=mask)` from `evaluate.py:102` raises `TypeError`.
    Line 143: `sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))` references `beta_t`, which is never assigned in `sample()`, throwing `NameError: name 'beta_t' is not defined`.

---

## 2. Logic Chain

1. **Premise 1 (Split unpacking)**: From Observation 1, `splitter.split()` returns 4 lists (`final_train_indices, val_indices, test_ind_indices, ood_indices`). Attempting to unpack 4 elements into 3 variables (`train_idx, val_idx, ood_idx`) in `main.py:54` causes Python to immediately raise `ValueError`. Thus, `main.py` cannot execute.
2. **Premise 2 (Inference launch failure)**: From Observation 7, `inference.py` hardcodes `checkpoint_epoch_22.pt`. Since the directory is empty, Python immediately raises `FileNotFoundError`. Furthermore, from Observation 8, even if a dummy file existed, the script blocks on `input()`, preventing automated execution. Thus, `inference.py` fails acceptance criteria R3.
3. **Premise 3 (Evaluation script crash)**: From Observation 12, `evaluate.py` mirrors Blueprint 2.3, but Blueprint 2.3 assumed an `AvatarDataset(data_dir=...)` constructor that does not exist in `preprocessing/dataset.py`. Running `evaluate.py` raises `TypeError`. Furthermore, from Observation 13, `evaluate.py` calls `sample(..., mask=mask)`, which `models/diffusion.py` rejects with `TypeError`. Thus, `evaluate.py` cannot execute.
4. **Premise 4 (Sampling math crash)**: From Observation 13, `models/diffusion.py:143` accesses `beta_t` without defining it. Because reverse diffusion iterates from $t=999$ down to $0$, step $t=999$ attempts to compute Langevin noise using `beta_t` and crashes with `NameError`. Thus, all reverse sampling loops (in `inference.py`, `evaluate.py`, or tests) fail at runtime.
5. **Premise 5 (Training baseline crash)**: From Observation 2, `train.py:130` requires `mask` when invoking `unet`, but the unconditional branch does not set `mask`. Thus, running the unconditional baseline experiment crashes with `UnboundLocalError`.
6. **Premise 6 (Blueprint implementation status)**:
   - Blueprint 2.2 (Metrics Range Normalization): Fully implemented in `metrics.py:65-88`.
   - Blueprint 2.3 (Evaluation Script): Created as `evaluate.py`, but blocked by constructor and method signature incompatibilities.
   - Blueprint 2.4 (Attribute Alignment Evaluator): Added to `metrics.py:131-139`, but not invoked in `evaluate.py`.
   - Blueprint 3.1 (Optimizer & Warmup): Decoupled clipping is present in `train.py:138-139`; optimizer parameter grouping and warmup are inlined in `main.py:98-125`; function abstraction is missing.
   - Blueprint 3.2 (Checkpoint Resolution): Partially written in `evaluate.py:17-28`; absent in `main.py` (uses `getctime`) and `inference.py` (hardcoded).
   - Blueprint 3.3 (CLI & DDIM Sampler): Unimplemented in `inference.py`.

---

## 3. Caveats

1. **Hardware / Runtime Constraints**: In benchmark mode, live execution of training runs (requiring GPU acceleration and dataset downloads) was not run directly in this survey turn. Findings are based on static forensic code analysis, signature verification, and call-graph tracing.
2. **Third-Party Metric Weights**: In `metrics.py`, `FrechetInceptionDistance`, `KernelInceptionDistance`, and `LearnedPerceptualImagePatchSimilarity` download Inception-v3 and VGG weights from torchvision/torchmetrics on first run, which requires network access if cached weights are not present.

---

## 4. Conclusion

1. **Current Codebase Operability**: None of the three top-level execution scripts (`main.py`, `inference.py`, `evaluate.py`) can currently run to completion without crashing due to:
   - `main.py`: 4-way split tuple unpack mismatch (`ValueError`).
   - `inference.py`: Hardcoded missing checkpoint path (`FileNotFoundError`) and blocking `input()` loop.
   - `evaluate.py`: Constructor mismatch with `AvatarDataset` (`TypeError`) and method parameter mismatch with `sample(..., mask=mask)` (`TypeError`).
   - `models/diffusion.py`: Undefined `beta_t` in `sample()` (`NameError`).
2. **Remediation Priority**:
   - Priority 1: Patch `models/diffusion.py:sample()` to accept `mask`/`uncond_mask` and compute `beta_t = self.betas[t].to(x.device)[:, None, None, None]`.
   - Priority 2: Fix `main.py:54` to unpack all 4 splits (`train_idx, val_idx, test_ind_idx, ood_idx`) and instantiate validation loader.
   - Priority 3: Fix `evaluate.py:52` to load metadata and image paths via CSV rather than passing non-existent `data_dir`.
   - Priority 4: Implement Blueprint 3.3 in `inference.py` with `argparse`, DDIM fast sampling, and Blueprint 3.2 checkpoint resolution.
   - Priority 5: Fix `train.py:103-130` by setting `mask = None` when `conditional=False`.

---

## 5. Verification Method

To independently verify these findings, inspect the following exact file locations and run the specified verification checks:

1. **Unpacking Mismatch**:
   - Inspect `main.py` line 54: `train_idx, val_idx, ood_idx = splitter.split(raw_metadata)`
   - Inspect `preprocessing/splitter.py` line 47: `return final_train_indices, val_indices, test_ind_indices, ood_indices`
   - Invalidation: If line 54 unpacks 4 values, this finding is invalidated.

2. **Reverse Process `beta_t` NameError**:
   - Inspect `models/diffusion.py` lines 115-144. Observe that `beta_t` is used at line 143:
     `sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))`
     but is never assigned within the `sample()` function.
   - Invalidation: If `beta_t = self.betas[t]...` is present before line 143, this finding is invalidated.

3. **`AvatarDataset` Constructor Mismatch in `evaluate.py`**:
   - Inspect `evaluate.py` line 52: `dataset = AvatarDataset(data_dir=data_dir, config=config, tokenizer=tokenizer)`
   - Inspect `preprocessing/dataset.py` lines 16-22: `def __init__(self, image_paths, metadata, tokenizer, config):`
   - Invalidation: If `AvatarDataset` accepts `data_dir` or `evaluate.py` passes `image_paths` and `metadata`, this finding is invalidated.

4. **Missing Mask Argument in `sample()`**:
   - Inspect `models/diffusion.py` line 75: `def sample(self, model, x, t, context=None, uncond_context=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):`
   - Inspect `evaluate.py` lines 102-104: calls `sample(..., mask=mask, uncond_mask=uncond_mask, ...)`
   - Invalidation: If `sample()` in `models/diffusion.py` includes `mask=None, uncond_mask=None`, this finding is invalidated.

5. **`inference.py` Hardcoded Paths & Interactive Loop**:
   - Inspect `inference.py` line 138 (`checkpoint_path = "checkpoints/checkpoint_epoch_22.pt"`) and line 143 (`while True: input(...)`).
   - Invalidation: If `argparse` is implemented and checkpoint is dynamically resolved, this finding is invalidated.
