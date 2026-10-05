# Comprehensive Investigation Report: Training, Evaluation & Usability

**Survey Explorer 3** — Training, Evaluation & Usability  
**Date**: October 5, 2026  
**Target Repository**: `c:\Users\Admin\Desktop\avatar diffusion`  
**Scope**: `train.py`, `main.py`, `inference.py`, `metrics.py`, `evaluate.py`, and related modules in `models/` and `preprocessing/`.  
**Defects Investigated**: DEF-06, DEF-07, DEF-08, DEF-10, DEF-11, DEF-12, DEF-14, DEF-18, DEF-19, DEF-20.  
**Blueprints Compared**: 2.2, 2.3, 2.4, 3.1, 3.2, 3.3.

---

## 1. Executive Summary

A forensic examination of the training, evaluation, and inference pipelines reveals that while key portions of DEF-17, DEF-18, and DEF-19 have been partially integrated into `metrics.py`, `train.py`, and `main.py`, **multiple critical defects and interface mismatches currently prevent `main.py`, `inference.py`, and `evaluate.py` from executing without crashing**.

Specifically:
1. `main.py` crashes on launch at line 54 with `ValueError: too many values to unpack (expected 3, got 4)` because `CompositionalSplitter.split()` returns 4 splits while `main.py` only unpacks 3.
2. `inference.py` crashes immediately on launch with `FileNotFoundError` due to a hardcoded non-existent checkpoint (`checkpoint_epoch_22.pt`), lacks dynamic checkpoint resolution, blocks automation via interactive `input()`, and forces 1,000 steps without DDIM support.
3. `evaluate.py` exists (matching Blueprint 2.3 verbatim) but crashes with `TypeError` because it instantiates `AvatarDataset(data_dir=...)` which does not match the actual constructor `(image_paths, metadata, tokenizer, config)`, and calls `reverse_process.sample(..., mask=mask, uncond_mask=uncond_mask)` which `models/diffusion.py` rejects.
4. `models/diffusion.py:143` contains an unhandled `NameError: name 'beta_t' is not defined` in `sample()`, crashing any reverse sampling call where $t > 0$ and `noise_free=False`.
5. `train.py:130` passes `mask=mask` to `unet`, but if `conditional=False`, `mask` is undefined, throwing `UnboundLocalError`. Furthermore, `train.py` lacks validation evaluation, sample generation (DEF-10), and AMP training (DEF-14).

### Status Summary Table

| Target Module | Defect ID | Associated Blueprint | Implementation Status | Critical Blocker / Finding |
|---|---|---|---|---|
| `train.py` | DEF-19 | Blueprint 3.1 | **Partially Applied** | Decoupled clipping is present (lines 138-139). `configure_optimizers` not defined. |
| `train.py` | DEF-10 | N/A | **Unimplemented** | No validation loop, no `val_loader`, no visual sample generation. |
| `train.py` | DEF-11 | N/A | **Defective** | `conditional=False` crashes with `UnboundLocalError: mask`. |
| `train.py` | DEF-14 | N/A | **Unimplemented** | Pure FP32 training without PyTorch AMP (`autocast` / `GradScaler`). |
| `main.py` | DEF-09 / DEF-13 | Blueprint 1.1 | **CRITICAL CRASH** | Line 54 unpacks 3 items from 4-way split: `ValueError`. Discards validation/OOD data. |
| `main.py` | DEF-19 | Blueprint 3.1 | **Partially Applied** | Inlined decay/no-decay and warmup scheduler. Missing `configure_optimizers` function. |
| `main.py` | DEF-20 | Blueprint 3.2 | **Defective** | Line 132 still uses `os.path.getctime` for checkpoint lookup. |
| `inference.py` | DEF-08 | Blueprint 3.2 | **CRITICAL CRASH** | Line 138 hardcodes `checkpoints/checkpoint_epoch_22.pt` -> `FileNotFoundError`. |
| `inference.py` | DEF-12 | Blueprint 3.3 | **CRITICAL DEFECT** | Blocks on interactive `input()`; OOD evaluation block unreachable; forced 1,000 steps without DDIM. |
| `metrics.py` | DEF-18 | Blueprint 2.2 | **Fully Applied** | Independent `_ensure_zero_one_range` correctly normalizes real and fake tensors to $[0.0, 1.0]$. |
| `metrics.py` | DEF-07 | Blueprint 2.4 | **Applied / Unwired** | `AttributeAlignmentEvaluator` is defined but never called anywhere. |
| `metrics.py` | DEF-06 | Blueprint 2.3 | **Dead Code** | `DiffusionEvaluator` never imported in `train.py` or `main.py`. |
| `evaluate.py` | DEF-06 | Blueprint 2.3 | **CRITICAL CRASH** | Exists, but crashes on `AvatarDataset(data_dir=...)` and `sample(..., mask=mask)`. |
| `models/diffusion.py` | DEF-17 / DEF-04 | Blueprint 2.1 | **CRITICAL CRASH** | Line 143 references undefined `beta_t` (`NameError`); lacks `mask`/`uncond_mask` arguments. |

---

## 2. Examination of `train.py` and `main.py`

### 2.1 Mask Wiring and Forward Pass in `train.py`
In `train.py`, line 130 passes `mask` to `unet`:
```python
# train.py:129-130
# 5. Predizione del rumore target (epsilon-prediction)
predicted_noise = unet(noisy_images, timesteps, context, mask=mask)
```
- In `models/unet.py:50`, `Unet.forward(self, x, time, context, mask=None)` accepts `mask` and passes it to all 6 `SpatialCrossAttention` layers (`attn_inc`, `attn_down1`, `attn_down2`, `attn_bott1`, `attn_up1`, `attn_up2`).
- In `models/unet_parts.py:237-274`, `SpatialCrossAttention.forward` accepts `mask=None` and applies rank-adaptive reshaping to 4D boolean masks for `F.scaled_dot_product_attention`.

#### Identified Bugs in Mask Handling:
1. **UnboundLocalError on Unconditional Baseline (`conditional=False`)**:
   In `train.py:103-128`:
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
   ```
   When `conditional=False`, `mask` is **never assigned** in the `else` branch. At line 130, calling `unet(..., mask=mask)` raises:
   `UnboundLocalError: local variable 'mask' referenced before assignment`.
2. **CFG Dropout Mask Inconsistency**:
   In `train.py:112-122`:
   ```python
   drop_mask = torch.rand(batch_size, device=device) < cfg_drop_rate
   if drop_mask.any():
       uncond_context = get_unconditional_context(...)
       context = torch.where(drop_mask.unsqueeze(1).unsqueeze(2), uncond_context, context)
   ```
   When conditioning is dropped for a sample in the batch, `context` is replaced with `uncond_context`, but `mask` is **not updated**. The original text padding mask continues to be passed to `unet` for dropped samples. While all unconditional tokens are `<PAD>` (or null), passing an active token mask rather than an all-pad or all-valid mask creates conditioning asymmetry between training CFG dropout and inference CFG unconditional branches.

### 2.2 Gradient Clipping and Optimizer Configuration (DEF-19)
The user noted that DEF-19 was already applied. Forensic verification confirms:
- **Decoupled Gradient Clipping**: In `train.py:138-139`:
  ```python
  torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
  torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
  ```
  This is correctly present and decoupled, preventing U-Net from dominating gradient clipping over the text encoder.
- **Optimizer and Scheduler Construction**:
  In `main.py:98-125`:
  ```python
  decay_params = []
  no_decay_params = []
  for model in [unet, text_encoder]:
      for name, param in model.named_parameters():
          if not param.requires_grad:
              continue
          if param.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower():
              no_decay_params.append(param)
          else:
              decay_params.append(param)
  optim_groups = [
      {"params": decay_params, "weight_decay": 1e-4},
      {"params": no_decay_params, "weight_decay": 0.0}
  ]
  optimizer = optim.AdamW(optim_groups, lr=1e-4)

  warmup_epochs = 5
  total_epochs = 50
  warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
  cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
  scheduler = optim.lr_scheduler.SequentialLR(
      optimizer, 
      schedulers=[warmup_scheduler, cosine_scheduler], 
      milestones=[warmup_epochs]
  )
  ```
  This logic correctly implements Blueprint 3.1. However, Blueprint 3.1 specifies this as a modular, testable function `def configure_optimizers(unet, text_encoder, lr=1e-4, weight_decay=1e-4, warmup_epochs=5, total_epochs=50)`. Currently, it is inlined directly in `main.py` and not exposed in `train.py`.

### 2.3 Validation Loop and Baseline Handling (DEF-10, DEF-11)
- **Missing Validation Loop (`DEF-10`)**:
  - `train.py` takes no validation dataloader argument (`val_loader`).
  - No validation loss is evaluated or logged at the end of each epoch.
  - No early stopping or best-checkpoint tracking is implemented.
  - No sample image generation is performed during training to visually inspect reverse diffusion convergence.
- **Unconditional Baseline Handling (`DEF-11`)**:
  - `main.py:173` hardcodes `conditional=True`.
  - There is no command-line flag (e.g. `--conditional` or `--mode baseline`) to trigger an unconditional training run.
  - As noted above, setting `conditional=False` crashes `train.py` due to `UnboundLocalError`.

### 2.4 Automatic Mixed Precision (AMP) Training (DEF-14)
- In `train.py:82-145`, all tensor operations and backward passes run in pure `float32`.
- There is no `torch.amp.autocast('cuda')` or `torch.cuda.amp.GradScaler()`, causing slower training and higher memory consumption on GPU.

### 2.5 Critical 4-Way Split Unpack Crash in `main.py`
In `main.py:53-54`:
```python
splitter = CompositionalSplitter(config)
train_idx, val_idx, ood_idx = splitter.split(raw_metadata)
```
- In `preprocessing/splitter.py:47`, Blueprint 1.1 was applied and returns:
  `return final_train_indices, val_indices, test_ind_indices, ood_indices` (4 elements).
- `main.py` attempts to unpack into 3 variables (`train_idx, val_idx, ood_idx`).
- **Impact**: Running `python main.py` immediately crashes with:
  `ValueError: too many values to unpack (expected 3, got 4)`.
- Furthermore, `val_idx` and `ood_idx` are completely discarded. Only `train_idx` is used to build `train_dataset`.

### 2.6 Checkpoint Resolution in `main.py` (DEF-20)
In `main.py:132`:
```python
latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
```
- Still uses `os.path.getctime`, which is non-portable across platforms and archive extractions. Blueprint 3.2 (regex sorting by integer epoch) was not applied to `main.py`.

---

## 3. Examination of `inference.py`

### 3.1 Hardcoded Paths and Crash on Launch (DEF-08)
In `inference.py:136-139`:
```python
generator = AvatarGenerator(
    config_path="./preprocessing/preprocessing_config.json", 
    checkpoint_path="checkpoints/checkpoint_epoch_22.pt"
)
```
- The `checkpoints/` directory is empty.
- When `AvatarGenerator` is initialized, line 35 calls `self._load_checkpoint("checkpoints/checkpoint_epoch_22.pt")`, triggering an unhandled `FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/checkpoint_epoch_22.pt'`.
- In `inference.py:24`, `self.tokenizer.load_vocab()` loads `preprocessing/vocab.json`. On a fresh repository before training/preprocessing, `vocab.json` does not exist, triggering another `FileNotFoundError`.

### 3.2 Blocking Interactive `input()` Loop (DEF-12)
In `inference.py:143-157`:
```python
while True:
    user_input = input("\nInserisci un prompt per generare l'avatar (o 'exit' per uscire): ")
    if user_input.lower() == 'exit':
        break
    try:
        seed_input = input("Inserisci un seed numerico (es. 42): ")
        user_seed = int(seed_input)
    ...
```
- The script halts and waits for interactive console input.
- It cannot be executed in automated testing, scripts, headless servers, or Docker containers.

### 3.3 Dead / Unreachable OOD Testing Block (DEF-12)
In `inference.py:160-164`:
```python
ood_test_prompts = [
    "a blue cartoon avatar with round eyes and exaggerated proportions",
    "a red avatar with standard eyes and normal proportions"
]
generator.evaluate_ood_combinations(ood_test_prompts, num_seeds=4)
```
- This code is positioned immediately after the infinite `while True:` loop.
- It will never execute unless a human user manually types `'exit'` into the console.
- In addition, the prompts use natural language tokens (`"blue"`, `"round"`, `"exaggerated"`) which do not exist in the training vocabulary (DEF-05).

### 3.4 Fixed 1,000-Step Reverse Process & Lack of DDIM Support (DEF-12)
In `inference.py:33` and line 75:
```python
self.reverse_process = DiffusionReverseProcess(num_time_steps=1000, device=self.device)
...
for t_step in reversed(range(self.reverse_process.num_time_steps)):
```
- Reverse diffusion strictly forces all 1,000 steps of standard DDPM sampling.
- Because CFG executes dual forward passes per step, generating a single avatar requires 2,000 U-Net forward evaluations.
- There is no support for an accelerated deterministic DDIM sampler (Song et al., 2020) or a configurable `--num_steps` parameter (e.g. 50 steps per Blueprint 3.3).

### 3.5 Checkpoint Resolution Logic (DEF-20)
- `inference.py` has no `resolve_checkpoint` function. It accepts whatever string is passed to `AvatarGenerator`, which defaults to the missing `checkpoint_epoch_22.pt`.
- Blueprint 3.2 (`resolve_checkpoint`) is entirely absent from `inference.py`.

---

## 4. Examination of `metrics.py`

### 4.1 Range Normalization (`_ensure_zero_one_range`) (DEF-18)
In `metrics.py:65-88`:
```python
@staticmethod
def _ensure_zero_one_range(images: torch.Tensor) -> torch.Tensor:
    """
    Normalizza in modo indipendente un tensore di immagini a [0.0, 1.0].
    Gestisce in modo sicuro input sia in [-1.0, 1.0] che in [0.0, 1.0],
    applicando clamp rigoroso per prevenire overflow in Inception-v3.
    """
    if images.min() < 0.0:
        images = (images + 1.0) / 2.0
    return torch.clamp(images, 0.0, 1.0)

def update_quality_metrics(self, real_images: torch.Tensor, fake_images: torch.Tensor):
    real_norm = self._ensure_zero_one_range(real_images)
    fake_norm = self._ensure_zero_one_range(fake_images)

    self.fid.update(real_norm, real=True)
    self.fid.update(fake_norm, real=False)
    
    self.kid.update(real_norm, real=True)
    self.kid.update(fake_norm, real=False)
```
- **Verdict**: **VERIFIED & COMPLIANT**. Blueprint 2.2 is 100% applied. Real and fake image tensors are now independently normalized to $[0.0, 1.0]$ and clamped, preventing the asymmetric range compression bug where fake images were mapped to $[0.5, 1.0]$.

### 4.2 Attribute Alignment Evaluator (`AttributeAlignmentEvaluator`) (DEF-07)
In `metrics.py:131-139`:
```python
class AttributeAlignmentEvaluator:
    def __init__(self, classifier_model, device):
        self.classifier = classifier_model.to(device).eval()

    def evaluate_alignment(self, images, expected_attributes):
        with torch.no_grad():
            preds = self.classifier(images)
            correct = (preds == expected_attributes).float().mean()
        return correct.item()
```
- **Verdict**: **APPLIED BUT UNWIRED**. The class matches Blueprint 2.4, but it is not imported or utilized in `evaluate.py`, `train.py`, or `main.py`.

### 4.3 Orphaned Metrics Suite (DEF-06)
- `DiffusionEvaluator` is never imported or called in `train.py`, `main.py`, or `inference.py`.
- While `evaluate.py` imports `DiffusionEvaluator`, `evaluate.py` itself has multiple fatal bugs preventing execution (see Section 5 below).

---

## 5. Examination of `evaluate.py`

### 5.1 Status of `evaluate.py`
`evaluate.py` **already exists** in the root directory (130 lines). Its contents mirror Blueprint 2.3 from `audit_report.md`.

### 5.2 Critical Defects and Integration Failures in `evaluate.py`

#### Defect A: `AvatarDataset` Signature Mismatch
In `evaluate.py:52`:
```python
dataset = AvatarDataset(data_dir=data_dir, config=config, tokenizer=tokenizer)
```
However, in `preprocessing/dataset.py:16-22`:
```python
def __init__(
    self,
    image_paths: List[str],
    metadata: List[Dict[str, Any]],
    tokenizer: AvatarTokenizer,
    config: PreprocessingConfig
):
```
`AvatarDataset` **does not accept `data_dir`**.
When `python evaluate.py` is invoked, it crashes immediately at line 52:
```
TypeError: AvatarDataset.__init__() got an unexpected keyword argument 'data_dir'
```
To construct `AvatarDataset`, the script must load `image_paths` and `raw_metadata` from the CSV file (`data/meta/cartoon_image_attributes.csv`) and image directory (`data/cartoonset100k_jpg`), exactly as `main.py:31-51` does.

#### Defect B: `reverse_process.sample()` Mask Parameter Rejection
In `evaluate.py:100-105`:
```python
x = reverse_process.sample(
    model=unet, x=x, t=t,
    context=cond_ctx, uncond_context=uncond_ctx,
    mask=mask, uncond_mask=uncond_mask,
    guidance_scale=3.5, clip_denoised=True
)
```
However, in `models/diffusion.py:75`:
```python
def sample(self, model, x, t, context=None, uncond_context=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):
```
`sample()` **does not accept `mask` or `uncond_mask`**.
Passing `mask` and `uncond_mask` causes Python to crash with:
```
TypeError: DiffusionReverseProcess.sample() got an unexpected keyword argument 'mask'
```
Blueprint 2.1 in `audit_report.md` included `mask` and `uncond_mask` in `sample()`, but `models/diffusion.py` was never updated to implement that signature!

#### Defect C: Undefined `beta_t` Variable in `models/diffusion.py:143`
In `models/diffusion.py:143`:
```python
sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))
```
`beta_t` is **never assigned** in `DiffusionReverseProcess.sample()`.
In Blueprint 2.1 line 1060:
`beta_t = self.betas[t].to(x.device)[:, None, None, None]`
This assignment was accidentally omitted from `models/diffusion.py`. Any call to `reverse_process.sample` with $t > 0$ and `noise_free=False` crashes with:
```
NameError: name 'beta_t' is not defined
```

#### Defect D: Missing File Dependencies on Clean Startup
- Line 38: `tokenizer.load_vocab()` crashes if `preprocessing/vocab.json` has not yet been generated.
- Line 55: `with open(splits_path, "r") as f: splits = json.load(f)` crashes if `splits.json` does not exist.
- Line 128: `resolve_checkpoint()` crashes with `FileNotFoundError` if `checkpoints/` contains no `.pt` files.

---

## 6. Line-by-Line Blueprint Comparison

### 6.1 Blueprint 2.2: Independent Metric Range Normalization (`metrics.py`)
| Blueprint 2.2 Specification | Current Code (`metrics.py:65-88`) | Match Verdict |
|---|---|---|
| `@staticmethod def _ensure_zero_one_range(images)` | Present at line 65 | **100% Match** |
| `if images.min() < 0.0: images = (images + 1.0) / 2.0` | Present at line 72-73 | **100% Match** |
| `return torch.clamp(images, 0.0, 1.0)` | Present at line 74 | **100% Match** |
| `real_norm = self._ensure_zero_one_range(real_images)` | Present at line 80 | **100% Match** |
| `fake_norm = self._ensure_zero_one_range(fake_images)` | Present at line 81 | **100% Match** |
| `self.fid.update(real_norm, real=True); self.fid.update(fake_norm, real=False)` | Present at lines 83-84 | **100% Match** |
| `self.kid.update(real_norm, real=True); self.kid.update(fake_norm, real=False)` | Present at lines 86-87 | **100% Match** |

**Conclusion**: Blueprint 2.2 is completely and accurately implemented in `metrics.py`.

---

### 6.2 Blueprint 2.3: Standalone Batch-Accumulating Evaluation Pipeline (`evaluate.py`)
| Blueprint 2.3 Specification | Current Code (`evaluate.py`) | Match Verdict | Discrepancy / Defect |
|---|---|---|---|
| `resolve_checkpoint(checkpoint_dir="checkpoints")` | Present at lines 17-28 | **Identical** | Does not accept `explicit_path` parameter. |
| `evaluate(checkpoint_path, data_dir, batch_size, num_samples)` | Present at lines 30-120 | **Identical** | Verbatim copy of Blueprint 2.3. |
| `dataset = AvatarDataset(data_dir=data_dir, ...)` | Present at line 52 | **Identical to BP** | **Blueprint Flaw**: Incompatible with `AvatarDataset(image_paths, metadata, tokenizer, config)`. Crashes on call. |
| `splits = json.load(f)` for `test_ind` and `test_ood` | Present at lines 55-61 | **Identical** | Relies on `preprocessing/splits.json` existing. |
| `reverse_process.sample(..., mask=mask, uncond_mask=uncond_mask, clip_denoised=True)` | Present at lines 100-105 | **Identical to BP** | **Interface Mismatch**: `models/diffusion.py:sample()` rejects `mask` and `uncond_mask`. Crashes on call. |
| `evaluator.update_quality_metrics(real_images, fake_images)` | Present at line 110 | **Identical** | Correctly accumulates batches before `compute_quality_metrics()`. |
| CLI `argparse` with `--checkpoint`, `--batch_size`, `--num_samples` | Present at lines 122-130 | **Identical** | Works when valid checkpoint exists. |

**Conclusion**: Blueprint 2.3 was copied directly into `evaluate.py`, but Blueprint 2.3 contains an internal defect (calling `AvatarDataset(data_dir=...)`) and depends on Blueprint 2.1 mask wiring in `models/diffusion.py` that was never applied.

---

### 6.3 Blueprint 2.4: Text-Image Alignment / Attribute Consistency Metric (`metrics.py`)
| Blueprint 2.4 Specification | Current Code (`metrics.py:131-139`) | Match Verdict |
|---|---|---|
| `class AttributeAlignmentEvaluator:` | Present at line 131 | **100% Match** |
| `def __init__(self, classifier_model, device): self.classifier = classifier_model.to(device).eval()` | Present at lines 132-133 | **100% Match** |
| `def evaluate_alignment(self, images, expected_attributes):` | Present at line 135 | **100% Match** |
| `with torch.no_grad(): preds = self.classifier(images); correct = (preds == expected_attributes).float().mean(); return correct.item()` | Present at lines 136-139 | **100% Match** |

**Conclusion**: Blueprint 2.4 is completely present in `metrics.py`. However, it is never instantiated or evaluated in `evaluate.py`.

---

### 6.4 Blueprint 3.1: Decoupled Parameter Optimizer & LR Warmup (`train.py` / `main.py`)
| Blueprint 3.1 Specification | Current Code (`main.py` & `train.py`) | Match Verdict | Discrepancy |
|---|---|---|---|
| `def configure_optimizers(unet, text_encoder, lr=1e-4, weight_decay=1e-4, warmup_epochs=5, total_epochs=50):` | **Not defined as function** in either `main.py` or `train.py`. | **Missing abstraction** | Inlined directly into `main.py:98-125`. |
| Parameter separation: `param.ndim <= 1 or "norm" in name or "bias" in name` -> `weight_decay=0.0` | Present in `main.py:98-114` | **Inlined Match** | Parameter logic matches Blueprint. |
| `LinearLR(start_factor=0.1, total_iters=5)` + `CosineAnnealingLR(T_max=45)` + `SequentialLR` | Present in `main.py:116-125` | **Inlined Match** | Scheduler setup matches Blueprint. |
| `torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)` | Present in `train.py:138` | **100% Match** | Correctly decoupled in training loop. |
| `torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0)` | Present in `train.py:139` | **100% Match** | Correctly decoupled in training loop. |

**Conclusion**: Decoupled gradient clipping is active in `train.py`. The optimizer and scheduler parameter logic is inlined in `main.py`, but missing the modular `configure_optimizers` helper function specified by Blueprint 3.1.

---

### 6.5 Blueprint 3.2: Deterministic Checkpoint Resolution (`inference.py` / `evaluate.py` / `main.py`)
| Blueprint 3.2 Specification | `inference.py` | `main.py` | `evaluate.py` | Verdict |
|---|---|---|---|---|
| `resolve_checkpoint(checkpoint_dir, explicit_path)` | **Missing** | **Missing** | Partially present (no `explicit_path`) | **NOT UNIFIED** |
| Deterministic integer regex sorting `checkpoint_epoch_(\d+)\.pt` | **Missing** | **Missing** (uses `os.path.getctime`) | Present at line 19 | **Missing in 2 of 3 scripts** |
| Fallback to `glob("*.pt")` | **Missing** | Present (line 128) | Present (line 24) | Inconsistent |

**Conclusion**: Blueprint 3.2 has only been implemented locally in `evaluate.py`. `main.py` still relies on `os.path.getctime`, and `inference.py` hardcodes an arbitrary path without resolution.

---

### 6.6 Blueprint 3.3: CLI Arguments & Accelerated DDIM Sampler (`inference.py`)
| Blueprint 3.3 Specification | Current Code (`inference.py`) | Match Verdict | Discrepancy |
|---|---|---|---|
| `import argparse` and `def parse_args():` | **Missing** | **Not Applied** | Still uses interactive `while True: input(...)`. |
| `--prompt` argument (default template prompt) | **Missing** | **Not Applied** | Prompts entered manually via `input()`. |
| `--seed` argument (default 42) | **Missing** | **Not Applied** | Seed entered manually via `input()`. |
| `--guidance_scale` argument (default 3.5) | **Missing** | **Not Applied** | Hardcoded to 3.5 in interactive loop. |
| `--checkpoint` argument (default None) | **Missing** | **Not Applied** | Hardcoded to non-existent epoch 22. |
| `--num_steps` argument (default 50) | **Missing** | **Not Applied** | Forced 1,000 steps in reverse loop. |
| Accelerated DDIM Sampler ($S=50$ steps) | **Missing** | **Not Applied** | Only standard 1,000-step DDPM loop in `DiffusionReverseProcess`. |

**Conclusion**: Blueprint 3.3 has not been implemented at all in `inference.py`.

---

## 7. Concrete Remediation Proposals for Implementers

To achieve zero-regression compliance and enable all scripts to run cleanly without exceptions, the following modifications are recommended for subsequent implementation:

### Proposal 1: Fix `models/diffusion.py` Signature & Undefined `beta_t`
1. In `models/diffusion.py:sample()`, accept `mask=None` and `uncond_mask=None`.
2. Concatenate `mask_input` when performing dual CFG forward passes, and pass `mask` to `model(...)`.
3. At step 2, assign `beta_t = self.betas[t].to(x.device)[:, None, None, None]` to fix the `NameError` crash at line 143.
4. Add a DDIM sampling method `sample_ddim()` or support a sub-sequence of timesteps when `num_steps < 1000`.

### Proposal 2: Fix `main.py` Unpack Bug & Checkpoint Resolution
1. Line 54: Change unpacking from `train_idx, val_idx, ood_idx` to:
   ```python
   train_idx, val_idx, test_ind_idx, ood_idx = splitter.split(raw_metadata)
   ```
2. Build a validation dataset and DataLoader from `val_idx` and pass it to `train()` to resolve DEF-10.
3. Replace line 132 `os.path.getctime` with deterministic epoch regex resolution from Blueprint 3.2.
4. Add CLI arguments (`argparse`) to allow running the unconditional baseline (`--conditional False`) to resolve DEF-11.

### Proposal 3: Fix `train.py` Mask Unbound Error & Add Validation Monitoring
1. When `conditional=False`, explicitly set `mask = None`.
2. Update the call to `unet`:
   ```python
   predicted_noise = unet(noisy_images, timesteps, context, mask=mask)
   ```
3. When CFG drops context for dropped samples, zero out or adjust `mask` for those batch rows.
4. Add an optional `val_loader` and calculate validation loss at the end of each epoch.
5. Add PyTorch AMP `torch.amp.autocast('cuda')` with `GradScaler` for 2-3x training speedup (DEF-14).

### Proposal 4: Refactor `inference.py` (Implement Blueprint 3.2 & 3.3)
1. Add `parse_args()` using `argparse` with `--prompt`, `--seed`, `--guidance_scale`, `--checkpoint`, `--num_steps`, and `--batch_size`.
2. Use `resolve_checkpoint()` to locate the latest valid checkpoint, or provide a mock/dry-run message if none exists.
3. Remove the blocking interactive `while True: input(...)` loop.
4. Make OOD inspection a CLI flag (e.g. `--eval_ood`) rather than unreachable dead code.
5. If `vocab.json` is missing, auto-fit or initialize tokenizer safely rather than crashing.

### Proposal 5: Fix `evaluate.py` Dataset Constructor & Execution Pipeline
1. In `evaluate.py:52`, replace `AvatarDataset(data_dir=...)` with:
   ```python
   # Load image paths and metadata from CSV
   csv_path = "./data/meta/cartoon_image_attributes.csv"
   image_dir = "./data/cartoonset100k_jpg"
   raw_metadata, image_paths = [], []
   with open(csv_path, mode='r', encoding='utf-8') as f:
       reader = csv.DictReader(f)
       for row in reader:
           raw_filename = row.pop('filename')
           image_paths.append(os.path.join(image_dir, raw_filename))
           raw_metadata.append(row)
   dataset = AvatarDataset(image_paths=image_paths, metadata=raw_metadata, tokenizer=tokenizer, config=config)
   ```
2. In `evaluate.py`, handle cases where `splits.json` or `checkpoints/` does not yet exist by generating splits or graceful error reporting.
3. Wire `AttributeAlignmentEvaluator` to measure text-image alignment.
