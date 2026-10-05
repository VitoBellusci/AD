# Comprehensive Codebase Pipeline Audit Report

**Auditor Agent**: `explorer_codebase_pipeline_1`  
**Date**: October 5, 2026  
**Project**: Tiny Text-Conditioned Avatar Diffusion from Scratch  
**Academic Reference**: Politecnico di Bari — *Deep Learning (2026 VI)*, Instructor: Vito Walter Anelli, Ph.D. (`Deep_Learning_2026_VI 1.pdf` / `pdf_content.txt`)  
**Scope**: Data Pipeline (`preprocessing/*`), Training Pipeline (`train.py`, `main.py`), Evaluation & Metrics (`metrics.py`), and Inference & Entry Points (`inference.py`).

---

## Executive Summary & Audit Scorecard

This audit report delivers an exhaustive, line-by-line forensic investigation into the data pipeline, training loop, evaluation metrics, and inference mechanisms of the Avatar Diffusion codebase. 

### High-Level Findings:
1. **Critical Compositional Split Bug**: The compositional out-of-distribution (OOD) split logic in `preprocessing/splitter.py` fails silently. `preprocessing/preprocessing_config.json` filters by `("color", "blue")` and `("proportion", "exaggerated")`, attributes that **do not exist** in `data/meta/cartoon_image_attributes.csv`. Consequently, `ood_indices` is **100% empty (0 samples)**, defeating the core research question of the academic assignment.
2. **Missing Test & Discarded Validation Splits**: No ordinary test split is produced. The 10% validation split is extracted but immediately discarded in `main.py`—never wrapped into a DataLoader and never evaluated.
3. **Severe Tokenizer Vocabulary Index Collision**: `AvatarTokenizer.fit()` initializes its counter at `0`, causing the first three training vocabulary tokens to overwrite the special token IDs (`<UNK>: 1`, `<SOS>: 2`, `<EOS>: 3`). Furthermore, vocabulary mapping is based on synthetic integer attribute strings with trailing commas (e.g., `'1,'`, `'98,'`), creating a complete semantic disconnect with natural language prompts at inference.
4. **Attention Masking Failure (`-1e-9`)**: In `models/transformer.py:139`, masked tokens are filled with `-1e-9` instead of `-inf` or `-1e9`. Because $\exp(-10^{-9}) \approx 1.0$, padded `<PAD>` tokens receive almost identical attention weights to real tokens, corrupting self-attention and unconditional CFG representations.
5. **Completely Orphaned `metrics.py`**: While `DiffusionEvaluator` defines FID, KID, LPIPS diversity, and efficiency metrics, it is **never imported or executed anywhere** in the project. There is zero text-image alignment/conditioning metric implemented, and no evaluation script exists.
6. **Fragile Inference & Hardcoded Crashes**: `inference.py` relies on an interactive `input()` loop instead of CLI flags (`argparse`), hardcodes an epoch 22 checkpoint that does not exist in `checkpoints/`, crashes on missing `vocab.json`, lacks step reduction controls (forcing 1000 DDPM steps), and maps user prompts like `"a blue cartoon avatar..."` almost entirely to `<UNK>`.

| Component / Requirement | Status | Severity | Primary File & Lines |
|---|---|---|---|
| **Compositional OOD Split** | **FAILED (0 Samples)** | CRITICAL | `preprocessing/splitter.py:35-43`, `preprocessing_config.json:7-12` |
| **Ordinary Test Split** | **MISSING** | HIGH | `preprocessing/splitter.py:48-52` |
| **Validation Split Usage** | **DISCARDED** | HIGH | `main.py:54-57`, `train.py:78` |
| **Split Persistence / Caching** | **MISSING** | MEDIUM | `main.py:52-58` |
| **Tokenizer ID Assignment** | **CORRUPTED (Collisions)** | CRITICAL | `preprocessing/tokenizer.py:20-29` |
| **Prompt vs Vocab Semantics** | **DISCONNECTED** | CRITICAL | `caption_generator.py:10-13`, `inference.py:160-163` |
| **Text Encoder Pad Masking** | **BROKEN (`-1e-9`)** | HIGH | `models/transformer.py:139` |
| **Training Validation Loop** | **MISSING** | HIGH | `train.py:78-148` |
| **Unconditional Baseline Run**| **INCOMPLETE** | MEDIUM | `main.py:147`, `train.py:54-57` |
| **Mixed Precision (AMP)** | **MISSING (Pure FP32)**| MEDIUM | `train.py:82-140` |
| **Evaluation Metrics (`metrics.py`)** | **ORPHANED (Uncalled)** | HIGH | `metrics.py:8-121` |
| **Text-Image Alignment Metric** | **MISSING** | HIGH | `metrics.py:16-24` |
| **Evaluation Dataset Sampling** | **MISSING** | HIGH | Entire codebase |
| **Inference CLI / Automation** | **MISSING (Interactive)** | MEDIUM | `inference.py:143-157` |
| **Inference Step Control (DDIM)**| **MISSING (1000 steps)**| MEDIUM | `inference.py:75-89` |
| **Checkpoint Path in Inference** | **HARDCODED (Crash)** | HIGH | `inference.py:138` |

---

## Section 1: Dataset & Data Splits Deep-Dive

### 1.1 Dataset Loading, Preprocessing, Resizing, and Normalization
In `main.py:31-50` and `preprocessing/dataset.py:16-53`:
- **Image Source**: Google Cartoon Set (`data/cartoonset100k_jpg`) metadata is loaded via `csv.DictReader` from `data/meta/cartoon_image_attributes.csv`. The `filename` field is stripped, and relative paths (`0/cs11556364481883459966.jpg`) are mapped to full file paths via `os.path.join(image_dir, raw_filename)`.
- **Image Preprocessing**:
  ```python
  self.transform = transforms.Compose([
      transforms.Resize(self.config.resolution), # [64, 64]
      transforms.ToTensor(),                     # Maps uint8 [0, 255] to float32 [0.0, 1.0]
      transforms.Normalize(
          mean=self.config.mean,                 # [0.5, 0.5, 0.5]
          std=self.config.std                    # [0.5, 0.5, 0.5]
      )
  ])
  ```
  This standardizes images to $[-1.0, 1.0]$, adhering to the required pixel-space DDPM input domain. Images are loaded as PIL RGB (`Image.open(img_path).convert("RGB")`).
- **Configuration**: Managed via `PreprocessingConfig` (`preprocessing/config.py:4-37`) loading `preprocessing/preprocessing_config.json`. It validates `norm_min < norm_max`.

### 1.2 Dataset Splitting & The Compositional OOD Bug
The assignment specification states:
> *"The main split must be defined over attribute combinations rather than only over individual images. Hold out a controlled set of combinations, for example a particular color–artwork or color–proportion combination, and ensure that the corresponding combination is absent from training. You must report: which individual attributes appear in training; which combinations are held out; the number of train, validation, ordinary test, and compositional-OOD examples..."*

In `preprocessing/splitter.py:27-46`:
```python
for idx, meta in enumerate(metadata_list):
    current_comb = { (k, v) for k, v in meta.items() }
    is_ood = False
    for blocked_set in self.config.ood_blocked_combinations:
        if blocked_set.issubset(current_comb):
            is_ood = True
            break

    if is_ood:
        ood_indices.append(idx)
    else:
        train_indices.append(idx)
```
In `preprocessing/preprocessing_config.json:7-12`:
```json
"ood_blocked_combinations": [
  [
    ["color", "blue"],
    ["proportion", "exaggerated"]
  ]
]
```
#### Forensic Observation:
1. `data/meta/cartoon_image_attributes.csv` contains columns:  
   `eye_angle, eye_lashes, eye_lid, chin_length, eyebrow_weight, eyebrow_shape, eyebrow_thickness, face_shape, facial_hair, hair, eye_color, face_color, hair_color, glasses, glasses_color, eye_slant, eyebrow_width, eye_eyebrow_distance`.
2. Every value is an integer stored as a string (e.g., `'eye_color': '4'`, `'face_color': '1'`).
3. Neither `'color'` nor `'proportion'` exists in `meta.keys()`.
4. Neither `'blue'` nor `'exaggerated'` exists in `meta.values()`.
5. Therefore, `blocked_set.issubset(current_comb)` evaluates to **`False` for all 100,000 samples**.
6. **Result**: `ood_indices` has length **0**. Not a single image is held out as OOD. All images pass into `train_indices`.
7. In `splitter.py:54-64`, `verify_ood_isolation()` also fails to detect anything because the check looks for the non-existent keys `("color", "blue")`, returning `True` trivially.

### 1.3 Missing Ordinary Test Split & Discarded Validation Split
In `preprocessing/splitter.py:48-52`:
```python
random.shuffle(train_indices)
val_indices = train_indices[:int(len(train_indices) * 0.1)]
train_indices = train_indices[int(len(train_indices) * 0.1):]
return train_indices, val_indices, ood_indices
```
1. **No Ordinary Test Set**: The splitter returns only 3 lists (`train_indices`, `val_indices`, `ood_indices`). The assignment requires four partitions: `train`, `validation`, `ordinary test`, and `compositional-OOD`. There is no test split generated.
2. **Validation Split Discarded in `main.py:53-77`**:
   ```python
   splitter = CompositionalSplitter(config)
   train_idx, val_idx, ood_idx = splitter.split(raw_metadata)
   train_metadata = [raw_metadata[i] for i in train_idx]
   train_image_paths = [image_paths[i] for i in train_idx]
   # ...
   train_dataset = AvatarDataset(...)
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
   ```
   `val_idx` and `ood_idx` are completely ignored. No `val_dataset` or `val_loader` is initialized.

### 1.4 Split Persistence & Reproducibility Violation
- `random.shuffle(train_indices)` in `splitter.py:48` relies on Python's global `random` state.
- Even though `set_seed(42)` is called at `main.py:23`, neither `main.py` nor `splitter.py` saves the resulting split indices to disk (e.g., `train_indices.json`, `val_indices.json`, `ood_indices.json`).
- If data order changes or files are loaded differently, splits cannot be reliably reproduced. This violates the assignment directive: *"Save the split definition and preprocessing configuration so that the complete pipeline can be reproduced."*

### 1.5 Tokenization, Vocabulary Indexing Bug, and Semantic Disconnect
In `preprocessing/tokenizer.py:9-30`:
```python
class AvatarTokenizer:
    def __init__(self, config=None):
        self.config = config
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def fit(self, training_texts: List[str]):
        word_count = 0
        for text in training_texts:
            tokens = text.lower().split()
            for token in tokens:
                if token not in self.vocab:
                    word_count += 1
                    self.vocab[token] = word_count

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
```

#### Forensic Observations:
1. **Critical Vocabulary Collision Bug**:
   - `self.vocab` is pre-populated with special tokens at keys `0, 1, 2, 3`.
   - `word_count` is initialized to `0`.
   - When the first unique word in the training set is encountered (e.g. `'avatar'`), `word_count += 1` sets `word_count = 1`.
   - `self.vocab['avatar'] = 1`! This **overwrites ID 1**, which was assigned to `<UNK>`!
   - Next token `'with'`: `word_count = 2`, `self.vocab['with'] = 2` (collides with `<SOS>`).
   - Next token `'face'`: `word_count = 3`, `self.vocab['face'] = 3` (collides with `<EOS>`).
   - In `self.inverse_vocab`, ID 1 maps to `'avatar'`, ID 2 to `'with'`, ID 3 to `'face'`. Special tokens `<UNK>`, `<SOS>`, and `<EOS>` are eliminated from the reverse vocabulary.
   - Any unknown word encoded at inference yields token ID 1, which decodes to `'avatar'`.
2. **Punctuation Contamination**:
   - `CaptionGenerator` (`preprocessing/caption_generator.py:10-13`) creates captions using:
     `"avatar with face {face_color}, hair {hair}, eyes {eye_color}, glasses {glasses}, and facial hair {facial_hair}"`
   - Tokenization via `text.lower().split()` does not strip punctuation.
   - Tokens in vocabulary include trailing commas: `'4,'`, `'98,'`, `'11,'`.
   - The token for `facial_hair` does not have a comma (e.g., `'3'`), whereas `eye_color` has a comma (`'3,'`). They become two distinct vocabulary entries!
   - Prompts provided without commas at inference time map entirely to `<UNK>`.
3. **Severe Semantic Disconnect**:
   - Captions are formatted as raw integer numbers (e.g., `avatar with face 1, hair 98, eyes 4...`).
   - The assignment specifications (`Section 1`, `Figure 1`) state:
     > *"A user should be able to enter a prompt such as 'a blue cartoon avatar with round eyes and exaggerated proportions'..."*
   - And `inference.py:161` tests: `"a blue cartoon avatar with round eyes and exaggerated proportions"`.
   - **None of the words** `"blue"`, `"round"`, `"exaggerated"`, `"proportions"` exist in the training vocabulary! They are all mapped to `<UNK>` (which collides with ID 1). The model was trained purely on integer strings, making natural language prompts completely unintelligible to the text encoder.

### 1.6 Attention Masking Failure in Padded Tokens
In `preprocessing/tokenizer.py:38-44`:
- Fixed `max_seq_len = 20`. Captions with 14 tokens are padded with `<PAD>` (token 0).
- Special tokens `<SOS>` and `<EOS>` are never inserted into the sequence during `encode()`.
- In `models/transformer.py:138-140`:
  ```python
  if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)
  attention_scores = attention_scores.softmax(dim=-1)
  ```
  Filling with `-1e-9` ($-10^{-9} = -0.000000001$) is a major flaw. Since $\exp(-10^{-9}) \approx 0.999999999$, the attention logits for padded tokens are **not** suppressed. Padded tokens receive nearly identical softmax attention weights to real tokens. The proper value is `-1e9` or `float('-inf')`.

---

## Section 2: Training Loop & Optimization Deep-Dive (`train.py`)

### 2.1 Optimization, Learning Rate & Scheduler
In `main.py:98-99` and `train.py:43-44`:
- **Optimizer**: Joint `AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=1e-4, weight_decay=1e-4)`. Simultaneous optimization of U-Net denoiser and full Transformer text encoder.
- **Scheduler**: `CosineAnnealingLR(optimizer, T_max=50)`. Updated at the conclusion of every epoch (`scheduler.step()` in `train.py:146`).
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(text_encoder.parameters()), max_norm=1.0)` at line 138 protects against gradient explosion in both the Transformer encoder and U-Net cross-attention blocks.

### 2.2 Hyperparameters & Hardware Budget
- **Batch Size**: 32 (`main.py:76`), `drop_last=True`.
- **Epochs**: 50 (`main.py:143`).
- **Parameter Count / Budget**:
  - `FullTextEncoder`: 3 Transformer layers, $d_{\text{model}}=128$, 4 heads, $d_{\text{ff}}=256$, max_seq_len=20 $\approx 0.35\text{M}$ parameters.
  - `Unet`: base channels 64, 2 downsamplings (64 $\rightarrow$ 128 $\rightarrow$ 256), spatial self-attention and cross-attention $\approx 4.2\text{M}$ parameters.
  - Total parameter count is $\approx 4.5\text{M}$ parameters, well within the "Tiny" budget (<10M parameters).

### 2.3 Loss Calculation & DDPM Objective
In `train.py:94-134`:
- **Timestep Sampling**: Uniform integer sampling over $t \in [0, T-1]$ via `torch.randint(0, forward_process.num_time_steps, (batch_size,), device=device).long()`.
- **Forward Noising**:
  `noisy_images = forward_process.add_noise(images, noise, timesteps)` using closed-form:
  $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
- **Objective**: Standard $\epsilon$-prediction DDPM objective:
  ```python
  predicted_noise = unet(noisy_images, timesteps, context)
  loss = criterion(predicted_noise, noise) # nn.MSELoss()
  ```
  Matches the theoretical formulation from Ho et al. (2020).

### 2.4 Total Absence of Validation During Training
In `train.py:78-148`:
- The outer epoch loop processes `dataloader` (training set only).
- **Zero validation passes occur**:
  - `train()` does not take `val_loader` or `val_dataset`.
  - No validation loss is evaluated or logged.
  - No sample images are synthesized during training to monitor generation fidelity.
  - No early stopping or validation-based checkpointing is possible.

### 2.5 Unconditional Baseline Experiment Handling
Assignment Section 7 specifies:
> *"The minimum required experiments are: unconditional neural baseline; conditional model with selected conditioning mechanism."*

In `train.py:49-57, 103-128`:
- `train()` has a boolean parameter `conditional: bool = True`.
- When `conditional = False`, all batches use `get_unconditional_context()` (empty/PAD tokens).
- However, `main.py:147` hardcodes `conditional=True`.
- There is no runner, CLI flag, or script configured to execute the unconditional baseline, save its baseline checkpoints, or compare its metrics against the conditional run.

### 2.6 Conditioning Dropout for Classifier-Free Guidance (CFG)
In `train.py:109-122`:
```python
if cfg_drop_rate > 0.0:
    drop_mask = torch.rand(batch_size, device=device) < cfg_drop_rate
    if drop_mask.any():
        uncond_context = get_unconditional_context(
            text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
        )
        context = torch.where(drop_mask.unsqueeze(1).unsqueeze(2), uncond_context, context)
```
- `cfg_drop_rate = 0.1` drops text conditioning with 10% probability, enabling joint conditional/unconditional training for Classifier-Free Guidance.
- **Flaw in Unconditional Context**: `get_unconditional_context()` fills an all-PAD tensor. Because the attention mask in `models/transformer.py:139` uses `-1e-9`, the all-PAD sequence undergoes unmasked uniform attention rather than a zeroed or null embedding representation.

### 2.7 Checkpointing & Resumption
In `train.py:152-164` and `main.py:101-131`:
- Saves full state dicts: `unet_state_dict`, `text_encoder_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `loss`, `random_rng_state`, `numpy_rng_state`, `torch_rng_state`, `torch_cuda_rng_state`.
- In `main.py`, automatic resumption loads the latest `.pt` file in `checkpoints/`.
- **Minor Bug**: While `torch_rng_state` and `torch_cuda_rng_state` are restored at lines 121-126, `random_rng_state` and `numpy_rng_state` are ignored.

### 2.8 Numerical Stability & Precision (FP32 vs AMP)
- The entire pipeline runs in 32-bit floating point (`torch.float32`).
- No mixed precision (`torch.cuda.amp.autocast`) or gradient scaling (`torch.cuda.amp.GradScaler`) is implemented.
- While FP32 avoids half-precision underflow, on a cloud GPU (e.g., Nvidia T4), FP32 consumes double the VRAM and increases training time by $2\times$ to $3\times$.

---

## Section 3: Evaluation & Metrics Deep-Dive (`metrics.py`)

### 3.1 Overview of Implemented Metrics
`metrics.py` implements a standalone class `DiffusionEvaluator` (`metrics.py:8-121`):
1. **Fréchet Inception Distance (FID)**: `FrechetInceptionDistance(feature=64, normalize=True)` via `torchmetrics.image.fid`.
2. **Kernel Inception Distance (KID)**: `KernelInceptionDistance(subset_size=50, normalize=True)` via `torchmetrics.image.kid`.
3. **Diversity Across Seeds**: `LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)` via `torchmetrics.image.lpip` calculating mean pairwise LPIPS across seeds.
4. **Computational Efficiency**: `compute_efficiency_metrics()` measures parameter count, peak VRAM usage via `torch.cuda.max_memory_allocated()`, and sampling latency.

### 3.2 Evaluation Rules vs Pretrained Constraints
- **From-Scratch Rule Compliance**: The assignment strictly forbids pretrained models for the *generative pipeline* (denoiser, text encoder, VAE, latent representations). Using pretrained feature extractors (Inception-v3 for FID/KID, VGG for LPIPS) is standard for *evaluation metrics*.
- **Feature Layer Choice (`feature=64`)**: Standard FID computes statistics on the 2048-dimensional final pooling layer of Inception-v3. `feature=64` uses an early intermediate convolutional feature map. While acceptable for $32\times32$ or $64\times64$ inputs to mitigate rank deficiency on small sample sizes, this deviation must be explicitly justified in the final project report.

### 3.3 Complete Absence of Text-Image Alignment / Conditioning Metrics
Assignment Section 7 specifies:
> *"Mandatory metrics: Your evaluation must include both quality and conditioning metrics: image quality metric such as FID or KID... diversity across seeds for the same prompt; parameter count, sampling time, and memory usage."*

- **Critical Gap**: In `metrics.py`, there is **no conditioning metric**.
- No CLIP score (or from-scratch attribute alignment metric).
- No auxiliary attribute classification accuracy measuring whether an avatar generated from prompt `"eyes: 4, glasses: 11"` actually possesses glasses and eye color 4.
- Because conditioning metrics are absent, the central research question—generalization to unseen combinations—cannot be quantitatively measured.

### 3.4 Orphaned Module: Zero Integration in Codebase
- Grepping the codebase reveals that `DiffusionEvaluator` is **never imported or instantiated** anywhere outside `metrics.py`:
  - `main.py` does not import `metrics.py`.
  - `train.py` does not import `metrics.py`.
  - `inference.py` does not import `metrics.py`.
- There is **no evaluation script** (e.g., `evaluate.py`, `benchmark.py`, `test.py`).
- No batch sample generation loop exists to synthesize test sets and pass real/fake pairs to `update_quality_metrics()`.
- The evaluation module is completely dead code.

---

## Section 4: Inference & Entry Points Deep-Dive (`inference.py`, `main.py`)

### 4.1 Usability, CLI Flags & User Experience
In `main.py:21-150` and `inference.py:133-165`:
- Neither script uses `argparse`, `click`, or `sys.argv`.
- In `main.py`, batch size, epochs, paths, and training parameters are hardcoded inside `main()`.
- In `inference.py:143-157`, execution drops into a blocking interactive `while True:` loop calling `input()`:
  ```python
  user_input = input("\nInserisci un prompt per generare l'avatar (o 'exit' per uscire): ")
  ```
  This prevents headless execution, batch script evaluation, or pipeline automation.

### 4.2 Fatal Checkpoint and Vocab Crashes
1. **Hardcoded Missing Checkpoint**:
   In `inference.py:138`:
   `checkpoint_path="checkpoints/checkpoint_epoch_22.pt"`
   The `checkpoints/` folder in the repository is empty. Running `python inference.py` immediately crashes with an uncaught `FileNotFoundError`.
2. **Missing Vocab File**:
   In `inference.py:24`:
   `self.tokenizer.load_vocab()`
   `preprocessing/vocab.json` is not committed. A fresh clone crashes immediately unless `main.py` has run through line 65.

### 4.3 Deterministic Seeding Behavior
In `inference.py:51-53`:
- `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)` are set inside `generate()`.
- Gaussian noise `torch.randn((1, 3, h, w))` and stochastic reverse noise $z$ are strictly reproducible across seeds.

### 4.4 CFG Scale Argument & Formulation
In `inference.py:46, 81-89` and `models/diffusion.py:88-105`:
- `guidance_scale` is exposed as an argument (defaults to `3.0`, passed as `3.5` in `main`).
- Combined forward pass:
  ```python
  x_input = torch.cat([x, x], dim=0)
  t_input = torch.cat([t, t], dim=0)
  context_input = torch.cat([context, uncond_context], dim=0)
  all_noise = model(x_input, t_input, context=context_input)
  eps_cond, eps_uncond = torch.chunk(all_noise, 2, dim=0)
  predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
  ```
  Correctly implements Classifier-Free Guidance extrapolation.

### 4.5 Inference Steps & Sampling Speed Controls
- `inference.py:75` hardcodes full reverse sampling across all 1000 steps:
  `for t_step in reversed(range(self.reverse_process.num_time_steps)):`
- There is no argument to adjust inference steps (e.g. 50 or 100 steps) and no DDIM (Denoising Diffusion Implicit Models) deterministic sampler implemented. Generating a single image requires 1,000 dual (CFG) forward evaluations of the U-Net.

### 4.6 Fatal Prompt-to-Vocabulary Semantic Mismatch
- `inference.py:160-163` sets up OOD prompts:
  `"a blue cartoon avatar with round eyes and exaggerated proportions"`
- In `inference.py:56`, `self.tokenizer.encode(prompt)` is called.
- Because `AvatarTokenizer` was trained on template integer tokens (`"avatar"`, `"with"`, `"face"`, `"1,"`, `"hair"`, `"98,"`...), words such as `"blue"`, `"round"`, `"exaggerated"`, `"proportions"` are absent.
- Every single descriptive keyword maps to `<UNK>` (which maps to token ID 1).
- The text encoder outputs an embedding of repeated token 1s. The conditioning signal is effectively erased.

### 4.7 Dead / Unreachable Code in `inference.py`
In `inference.py:143-164`:
- The OOD evaluation call `generator.evaluate_ood_combinations(ood_test_prompts, num_seeds=4)` at line 164 is located directly after the infinite `while True:` loop.
- Unless the interactive user explicitly discovers and inputs `'exit'`, the script never executes the OOD evaluation code.

---

## Section 5: Academic Assignment Traceability Matrix

| Section & Requirement in PDF | Codebase Implementation Status | Findings & Deviations |
|---|---|---|
| **Sec 1: Research Question** (Generalize to unseen attribute combinations) | **FAILED** | OOD split is empty (`len(ood_indices) == 0`). No held-out combinations tested. |
| **Sec 2: Preprocessing Pipeline** (Reproducible image & caption pipeline) | **PARTIAL** | Pipeline works for $64\times64$ RGB images, but split assignments are not saved to disk. |
| **Sec 2: Text Tokenizer & Encoder from Scratch** | **PARTIAL / BUGGY** | From scratch, no pretrained weights. But vocabulary collisions on IDs 1-3, and attention mask uses `-1e-9`. |
| **Sec 2: Compact U-Net from Scratch** | **COMPLIANT** | Fully custom U-Net with GroupNorm, time MLP, spatial cross-attention, $\approx 4.2\text{M}$ params. |
| **Sec 2: DDPM Process from Scratch** | **COMPLIANT** | Cosine schedule, forward noising, reverse sampling, $\epsilon$-prediction MSE loss. |
| **Sec 4: Compositional Split Definition** | **NON-COMPLIANT** | Filter keys `("color", "blue")` do not match CSV attributes. 0 OOD samples held out. |
| **Sec 4: Split Statistics Reporting** (Train, Val, Test, OOD counts) | **NON-COMPLIANT** | Only train and val created. Test is missing. OOD is empty. |
| **Sec 4: Mandatory From-Scratch Constraints** | **COMPLIANT** | Zero pretrained backbones (no SD, CLIP, T5, BERT, diffusers). |
| **Sec 4: Tokenizer derived from training split only** | **COMPLIANT** | `tokenizer.fit(train_texts)` only on training split. |
| **Sec 7: Minimum Experiments: Unconditional Baseline** | **INCOMPLETE** | Implemented as a code flag in `train.py`, but never run, saved, or benchmarked in `main.py`. |
| **Sec 7: Minimum Experiments: Conditional Model** | **COMPLIANT** | Implemented and configured in `main.py`. |
| **Sec 7: Mandatory Metrics: FID or KID** | **PARTIAL / ORPHANED**| Implemented in `metrics.py` (feature=64), but never invoked or connected to any dataset loop. |
| **Sec 7: Mandatory Metrics: Conditioning Metrics** | **MISSING** | No text-image alignment or attribute fidelity metric implemented. |
| **Sec 7: Mandatory Metrics: Diversity across seeds** | **PARTIAL / ORPHANED**| Pairwise LPIPS implemented in `metrics.py`, but never executed in evaluation. |
| **Sec 7: Mandatory Metrics: Efficiency (Params, Time, VRAM)**| **PARTIAL / ORPHANED**| Implemented in `DiffusionEvaluator`, never integrated into training or testing. |
| **Sec 8: CLI & Inspectability** (Enter prompt, seed, view image) | **PARTIAL / CRAGGY** | Interactive `input()` prompt exists, but hardcoded checkpoint path crashes and vocab mismatches prompt words. |

---

## Section 6: Prioritized Recommendations & Remediation Plan

### Priority 1: Critical Bug Fixes (Must Be Fixed to Function)
1. **Fix Compositional Split Keys in `preprocessing_config.json`**:
   - Change `ood_blocked_combinations` to match actual columns in `cartoon_image_attributes.csv`.
   - *Example*: Hold out avatars with `hair = "98"` and `glasses = "11"`, or `face_color = "1"` and `hair_color = "2"`. Ensure both attributes appear individually across the training set, but never simultaneously.
   - Update `CompositionalSplitter` to output 4 splits: `train`, `val`, `test_ordinary`, `test_ood`.
   - Save split indices to disk (`data/splits.json`) for exact reproducibility.
2. **Fix `AvatarTokenizer.fit()` Indexing**:
   - Initialize `word_count = len(self.vocab)` (i.e. starting from ID 4) so training words do not overwrite `<UNK>: 1`, `<SOS>: 2`, `<EOS>: 3`.
   - Strip punctuation during tokenization: `re.sub(r'[^\w\s]', '', text).lower().split()`.
3. **Align Caption Semantics with Natural Language Prompts**:
   - Either:
     - Map numerical attributes to descriptive words (e.g. mapping `face_color 1` to a color descriptor, `glasses 11` to `'round glasses'`, `hair 98` to `'spiky hair'`).
     - Or update `inference.py` so prompts use the exact structured syntax the model was trained on.
4. **Fix Attention Masking in `models/transformer.py:139`**:
   - Replace `-1e-9` with `-1e9` or `-torch.finfo(attention_scores.dtype).max` so padded positions are truly masked out in softmax.

### Priority 2: Evaluation Pipeline Integration
5. **Create a Dedicated Evaluation Script (`evaluate.py`)**:
   - Load trained checkpoints, test dataset (ordinary and OOD), and run batch sampling.
   - Feed real and generated images to `DiffusionEvaluator.update_quality_metrics()`.
   - Output quantitative FID, KID, LPIPS diversity, and efficiency stats to a formatted table or JSON report.
6. **Implement Attribute Conditioning / Alignment Metric**:
   - Build a lightweight attribute verification evaluator that checks whether images generated from given attributes display those visual features (e.g., using a small pretrained or from-scratch attribute classifier).

### Priority 3: Training & Inference Usability
7. **Add Validation Loop in `train.py`**:
   - Pass `val_loader` to `train()`. Compute validation MSE loss and log progress each epoch.
8. **Replace Interactive `input()` with Standard CLI Flags (`argparse`)**:
   - In `inference.py`: `--prompt`, `--seed`, `--checkpoint`, `--guidance_scale`, `--output_dir`.
   - Add graceful fallback if a checkpoint does not exist.
9. **Implement DDIM Sampler for Fast Inference**:
   - Add a DDIM reverse sampling step to allow generation in 50 steps instead of 1000 steps.
10. **Enable PyTorch AMP (Automatic Mixed Precision)**:
    - Add `torch.cuda.amp.autocast()` and `GradScaler()` to `train.py` for significant speed and memory improvements on T4 GPUs.
