# Handoff Report — Data Pipeline, Training Loop, Metrics, and Inference Audit

**Agent**: `explorer_codebase_pipeline_1`  
**Handoff Type**: Hard (Task Complete)  
**Target Recipient**: `orchestrator_1` (Conversation ID: `ba5d9ddf-6d54-409f-85c1-89453b677f70`)  
**Date**: 2026-10-05T14:15:00Z  

---

## 1. Observation

Direct code observations across the codebase:

### 1.1 Dataset & Splitting
- **`preprocessing/preprocessing_config.json:7-12`**:
  ```json
  "ood_blocked_combinations": [
    [
      ["color", "blue"],
      ["proportion", "exaggerated"]
    ]
  ]
  ```
- **`data/meta/cartoon_image_attributes.csv:1-2`**:
  `filename,eye_angle,eye_lashes,eye_lid,chin_length,eyebrow_weight,eyebrow_shape,eyebrow_thickness,face_shape,facial_hair,hair,eye_color,face_color,hair_color,glasses,glasses_color,eye_slant,eyebrow_width,eye_eyebrow_distance`
  `0/cs11556364481883459966.jpg,2,0,0,1,1,13,0,5,3,98,4,1,2,11,2,1,2,2`
- **`preprocessing/splitter.py:35-43`**:
  ```python
  for blocked_set in self.config.ood_blocked_combinations:
      if blocked_set.issubset(current_comb):
          is_ood = True
          break
  if is_ood:
      ood_indices.append(idx)
  else:
      train_indices.append(idx)
  ```
  `blocked_set` is `{('color', 'blue'), ('proportion', 'exaggerated')}`. Since keys `'color'` and `'proportion'` do not exist in `meta`, `is_ood` is never `True`. `ood_indices` has length 0.
- **`preprocessing/splitter.py:48-52`**:
  ```python
  random.shuffle(train_indices)
  val_indices = train_indices[:int(len(train_indices) * 0.1)]
  train_indices = train_indices[int(len(train_indices) * 0.1):]
  return train_indices, val_indices, ood_indices
  ```
  Only 3 splits returned. Ordinary test split is missing.
- **`main.py:53-76`**:
  `train_idx, val_idx, ood_idx = splitter.split(raw_metadata)`
  `train_metadata = [raw_metadata[i] for i in train_idx]`
  `val_idx` is never referenced again; no validation DataLoader is created.

### 1.2 Tokenizer & Attention Masking
- **`preprocessing/tokenizer.py:12, 20-27`**:
  ```python
  self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
  # ...
  word_count = 0
  for text in training_texts:
      tokens = text.lower().split()
      for token in tokens:
          if token not in self.vocab:
              word_count += 1
              self.vocab[token] = word_count
  ```
  `word_count` starts at 0, assigning IDs 1, 2, 3 to the first three words, colliding with `<UNK>`, `<SOS>`, and `<EOS>`.
- **`preprocessing/caption_generator.py:10-13`**:
  `template = "avatar with face {face_color}, hair {hair}, eyes {eye_color}, glasses {glasses}, and facial hair {facial_hair}"`
  Tokenized via `split()`, tokens include trailing commas: `'1,'`, `'98,'`, `'4,'`.
- **`models/transformer.py:138-140`**:
  ```python
  if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)
  attention_scores = attention_scores.softmax(dim=-1)
  ```
  Value is `-1e-9` ($-10^{-9}$), which leaves padded tokens unmasked in softmax.

### 1.3 Training Loop
- **`train.py:78-148`**: Loop executes `for epoch in range(start_epoch, epochs): for images, text_tokens in progress_bar: ... scheduler.step(); torch.save(...)`. No validation loop exists.
- **`main.py:147`**: Hardcodes `conditional=True`. No unconditional baseline experiment is run or benchmarked.
- **`train.py:82-140`**: Operates in pure FP32. No PyTorch AMP (`autocast` / `GradScaler`).

### 1.4 Metrics & Evaluation
- **`metrics.py:4-23`**:
  `DiffusionEvaluator` initializes `FrechetInceptionDistance(feature=64, normalize=True)`, `KernelInceptionDistance(subset_size=50, normalize=True)`, and `LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)`.
- **Grep check**: `DiffusionEvaluator` is never imported or called in `main.py`, `train.py`, `inference.py`, or anywhere else.
- No text-image alignment/conditioning metric is implemented.
- No evaluation script (`eval.py`, `test.py`) exists.

### 1.5 Inference
- **`inference.py:138`**:
  `checkpoint_path="checkpoints/checkpoint_epoch_22.pt"`
  Directory `checkpoints/` is empty. Running `python inference.py` fails with `FileNotFoundError`.
- **`inference.py:143-157`**: Interactive `while True: input(...)` blocks non-interactive script execution.
- **`inference.py:161`**: Prompt tested is `"a blue cartoon avatar with round eyes and exaggerated proportions"`. Because the training vocabulary consists of numerical tokens (`"1,"`, `"98,"`), descriptive words are not in `self.vocab` and map to `<UNK>`.
- **`inference.py:75`**: Reverse sampling hardcodes all 1000 steps without DDIM support.

---

## 2. Logic Chain

1. **Premise 1**: The assignment's central research question is whether the model can generalize to unseen combinations of attributes held out during training.
   - *Observation*: `preprocessing_config.json` looks for `("color", "blue")` and `("proportion", "exaggerated")`. The CSV metadata has columns like `eye_color`, `face_color`, `hair`, `glasses`, with numeric string values.
   - *Inference*: The subset condition `blocked_set.issubset(current_comb)` is mathematically impossible to satisfy. `len(ood_indices) == 0`.
   - *Deduction*: No compositional split exists at all; the model trains on the entire dataset without an OOD holdout, invalidating the core scientific experiment.

2. **Premise 2**: A text tokenizer must maintain distinct IDs for special tokens and map user vocabulary reliably.
   - *Observation*: `word_count` starts at 0, assigning 1 to `'avatar'`, 2 to `'with'`, 3 to `'face'`.
   - *Deduction*: Special tokens `<UNK>`, `<SOS>`, and `<EOS>` are overwritten. Reverse lookup fails.
   - *Observation*: Captions use numeric strings with commas (`'1,'`), while inference prompts use natural English (`'blue'`).
   - *Deduction*: Prompts at inference time map entirely to `<UNK>`, passing uninformative embeddings to the denoiser.

3. **Premise 3**: DDPM conditioning evaluation requires both quality (FID/KID) and conditioning alignment metrics across both ordinary and OOD test sets.
   - *Observation*: `metrics.py` contains FID, KID, LPIPS, and efficiency metrics, but is never called anywhere. It contains zero text-image alignment metrics. Test splits are absent.
   - *Deduction*: Quantitative evaluation cannot be executed as the codebase currently stands.

---

## 3. Caveats

- **Architecture Boundary**: In accordance with task boundaries, the internal layer implementation of `models/unet.py`, `models/unet_parts.py`, and `models/diffusion.py` was inspected primarily for interface alignment with the training loop and inference calls. Deeper architectural analysis belongs to `explorer_codebase_arch_1`.
- **External Dependencies**: `torchmetrics` requires internet access or cached weights to download Inception-v3 and VGG feature extractors on first run.

---

## 4. Conclusion

The pipeline exhibits severe, blocking disconnects between data preprocessing, training, evaluation, and inference:
1. **Compositional OOD split produces 0 samples** due to mismatched attribute keys.
2. **Tokenizer has vocabulary ID collisions** and a semantic mismatch with natural language prompts.
3. **Padded tokens are not masked** due to `-1e-9` in self-attention.
4. **Validation and ordinary test sets are omitted or discarded**.
5. **`metrics.py` is an orphaned module** with zero conditioning alignment metrics.
6. **`inference.py` crashes on launch** due to hardcoded non-existent checkpoint paths.

Remediation requires: aligning OOD configuration keys with CSV columns, fixing tokenizer index counters and text stripping, changing attention masking to `-1e9`, adding an ordinary test split and validation loop, creating a standalone `evaluate.py` script that invokes `DiffusionEvaluator`, and converting `inference.py` to use `argparse`.

---

## 5. Verification Method

To verify these findings independently:

1. **Verify Empty OOD Split**:
   Inspect `preprocessing/preprocessing_config.json:7-12` and compare against `data/meta/cartoon_image_attributes.csv:1`. Notice key mismatch (`color` vs `hair_color`/`eye_color`).
   Check `splitter.py:36` to verify subset condition.
2. **Verify Tokenizer Collision**:
   Inspect `preprocessing/tokenizer.py:12` and `preprocessing/tokenizer.py:20, 26-27`. Trace `word_count` starting at 0 for the first loop iterations.
3. **Verify Attention Masking Flaw**:
   Inspect `models/transformer.py:139`. Notice `masked_fill(mask == 0, -1e-9)`.
4. **Verify Orphaned Metrics**:
   Run grep for `DiffusionEvaluator` across project files (`main.py`, `train.py`, `inference.py`).
5. **Verify Inference Crash**:
   Inspect `inference.py:138` (`checkpoint_epoch_22.pt`) and check contents of `checkpoints/`.
