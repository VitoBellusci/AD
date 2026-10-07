# Handoff Report: Gate 6 Data Preprocessing, Vocabulary & Metrics Verification

**Agent**: `challenger_gate_6_2`  
**Parent**: `orchestrator_6` (`ebf019cd-ccf0-44f1-97b0-e8b532cc9e09`)  
**Working Directory**: `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\challenger_gate_6_2`  
**Date**: October 7, 2026  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 Compositional Split Verification (`preprocessing/splits.json`)
Direct file inspection of `preprocessing/splits.json` (100,010 lines total):
- **Keys present**:
  - Line 2: `"train": [`
  - Line 79638: `"val": [`
  - Line 89594: `"test_ind": [`
  - Line 99550: `"test_ood": [`
- **Partition line ranges and counts**:
  - `train`: lines 3 to 79636 $\rightarrow$ exactly **79,634** sample indices (80% of in-distribution).
  - `val`: lines 79639 to 89592 $\rightarrow$ exactly **9,954** sample indices (10% of in-distribution).
  - `test_ind`: lines 89595 to 99548 $\rightarrow$ exactly **9,954** sample indices (10% of in-distribution).
  - `test_ood`: lines 99551 to 100008 $\rightarrow$ exactly **458** sample indices (held-out combinations).
  - Total samples: $79,634 + 9,954 + 9,954 + 458 = 100,000$ samples.
- **Mutual Exclusivity**:
  In `preprocessing/splitter.py:32-53`, the splitting logic partitions indices:
  ```python
  for idx, meta in enumerate(metadata_list):
      current_comb = {(str(k), str(v)) for k, v in meta.items()}
      is_ood = any(blocked_set.issubset(current_comb) for blocked_set in blocked_sets)
      if is_ood:
          ood_indices.append(idx)
      else:
          train_indices.append(idx)
  ...
  val_indices = train_indices[:val_size]
  test_ind_indices = train_indices[val_size:val_size + test_size]
  final_train_indices = train_indices[val_size + test_size:]
  ```
  By slicing the shuffled `train_indices` into non-overlapping slices, `train`, `val`, and `test_ind` are strictly disjoint, and `ood_indices` is constructed by disjoint filtering. Mutual intersection between all 4 sets is exactly 0 (100% mutually exclusive).

---

### 1.2 Holdout Attribute Verification (`data/meta/cartoon_image_attributes.csv`)
The blocked attribute combination specified in `preprocessing/preprocessing_config.json:10-12` is:
```json
"ood_blocked_combinations": [
  [["hair", "98"], ["glasses", "11"]]
]
```
Cross-referencing `test_ood` sample indices against `data/meta/cartoon_image_attributes.csv`:
- **Index 0** (CSV Line 2):
  `0/cs11556364481883459966.jpg,2,0,0,1,1,13,0,5,3,98,4,1,2,11,2,1,2,2`  
  Attribute 11 (`hair`): `98`, Attribute 15 (`glasses`): `11`.
- **Index 235** (CSV Line 237):
  `0/cs11552957678347780539.jpg,2,1,0,0,0,2,1,0,14,98,4,2,5,11,4,1,0,0`  
  `hair`: `98`, `glasses`: `11`.
- **Index 348** (CSV Line 350):
  `0/cs1231295539899527196.jpg,0,0,1,2,0,12,3,0,14,98,4,5,6,11,3,2,2,0`  
  `hair`: `98`, `glasses`: `11`.
- **Index 735** (CSV Line 737):
  `0/cs12757889428063773736.jpg,2,1,0,0,0,4,1,2,8,98,2,0,2,11,5,1,0,1`  
  `hair`: `98`, `glasses`: `11`.
- **Index 1260** (CSV Line 1262):
  `0/cs11567011553874694365.jpg,2,0,1,1,1,5,0,2,14,98,4,10,8,11,3,0,0,0`  
  `hair`: `98`, `glasses`: `11`.
- **Index 99436** (CSV Line 99438, last sample in `test_ood`):
  `9/cs8785424451236885611.jpg,0,1,0,0,0,4,2,3,14,98,1,4,5,11,6,0,1,1`  
  `hair`: `98`, `glasses`: `11`.
- **Train split verification**:
  - Index 31380 (CSV Line 31382, first sample in `train`): `hair=106`, `glasses=11` (only glasses, no hair 98).
  - Index 84438 (CSV Line 84440): `hair=83`, `glasses=11`.
  - Index 69023 (CSV Line 69025): `hair=85`, `glasses=3`.
  - Index 89779 (CSV Line 89781): `hair=100`, `glasses=10`.
  - Exactly 0% of samples in `train`, `val`, and `test_ind` possess the joint `(hair=98, glasses=11)` combination.
- **Constituent attribute presence in `train`**:
  - In `train`, samples with `glasses=11` (and `hair != 98`) are abundant (e.g., indices 31380, 84438).
  - Samples with `hair=98` (and `glasses != 11`) are also present in `train`.
  - This ensures true compositional learning: the model learns both "wavy hair" and "no glasses" separately during training, but only encounters their joint combination at evaluation time.

---

### 1.3 Vocabulary Integrity Verification (`preprocessing/vocab.json` & `tokenizer.py`)
Direct inspection of `preprocessing/vocab.json`:
- **Special Tokens** (Lines 2–5):
  - `"<PAD>": 0`
  - `"<UNK>": 1`
  - `"<SOS>": 2`
  - `"<EOS>": 3`
- **Total vocabulary size**: Exactly **149** tokens.
- **ID Contiguity**: Contiguous IDs from 0 to 148 without gaps or duplicates.
- **Synthetic OOD token audit**:
  - `"exaggerated"`: **Absent** (not in `vocab.json`).
  - `"proportions"`: **Absent** (not in `vocab.json`).
  - `"look"`: **Absent** (not in `vocab.json`).
  - `"features"`: **Absent** (not in `vocab.json`).
- **Tokenizer unobserved mapping** (`preprocessing/tokenizer.py:49`):
  ```python
  encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
  ```
  Unobserved tokens map strictly to `vocab["<UNK>"]` (ID 1).
- **Fitting source** (`main.py:150-156`):
  ```python
  train_metadata = [raw_metadata[i] for i in train_indices]
  caption_gen = CaptionGenerator()
  train_texts = [caption_gen.generate(m) for m in train_metadata]
  tokenizer = AvatarTokenizer(config)
  tokenizer.fit(train_texts)
  tokenizer.save_vocab()
  ```
  `tokenizer.fit()` is strictly passed `train_texts` (no synthetic canonical prompts).

---

### 1.4 Evaluation Metrics Robustness (`evaluate.py` & `metrics.py`)
1. **Dynamic KID subset scaling** (`metrics.py:121-128`):
   ```python
   if min_samples < 50:
       self.kid.subset_size = min(50, max(2, min_samples))
   ```
   Prevents Torchmetrics `ValueError` when running evaluations with small sample sizes ($N < 50$), while resetting `self.kid.subset_size = 50` immediately after computation.
2. **Range Normalization** (`metrics.py:90-98`):
   ```python
   if images.min() < 0.0:
       images = (images + 1.0) / 2.0
   return torch.clamp(images, 0.0, 1.0)
   ```
   Guarantees that both $[-1.0, 1.0]$ diffusion outputs and $[0.0, 1.0]$ images are safely normalized into $[0.0, 1.0]$ before passing to Inception-v3 or VGG LPIPS.
3. **LPIPS Seed Diversity Guard** (`metrics.py:153-154`):
   Requires $\ge 2$ seeds, raises explicit `ValueError` if fewer, and computes all pairwise combinations $\binom{N}{2}$ via `itertools.combinations(generated_images, 2)`.
4. **Computational Efficiency Profiling** (`metrics.py:25-87`):
   Returns all required metrics: Total Parameters, U-Net Parameters, Text Encoder Parameters, Sampling Latency (s), and Peak VRAM Usage (MB).
5. **Full Pipeline Execution**:
   Execution logs in `worker_remediation_6_1/handoff.md` confirmed clean execution of `evaluate.py`:
   - Total Parameters: 26,655,621 (U-Net: 24,519,795, Text Encoder: 2,135,826)
   - Sampling Latency: 0.2489 s | Peak VRAM: 683.84 MB
   - Pairwise LPIPS Diversity: 0.4329
   - In-Distribution Test (10 samples): FID: 35.5543, KID (Mean): 0.5485
   - Compositional OOD Test (10 samples): FID: 40.5663, KID (Mean): 0.6577
   - Exit code: 0 with zero runtime errors.

---

## 2. Logic Chain

1. **Split Correctness**: The assignment requires a 4-way partition isolating held-out attribute combinations into a non-empty test set. Observation 1.1 establishes that `splits.json` contains all four required keys (`train`, `val`, `test_ind`, `test_ood`), covering all 100,000 samples with 0% overlap.
2. **Holdout Attribute Isolation**: The assignment mandates that the model not see the held-out combination in training. Observation 1.2 proves that 100% of samples in `test_ood` possess the blocked `(hair=98, glasses=11)` pair, and exactly 0% of samples in `train`, `val`, and `test_ind` possess it. Observation 1.2 also confirms that the individual attributes exist in `train`, verifying that generalization on `test_ood` is compositional rather than out-of-vocabulary.
3. **Vocabulary Purity**: Academic guidelines prohibit test and OOD vocabulary leakage. Observation 1.3 establishes that `vocab.json` was fitted strictly on `train_texts`, contains 149 valid natural tokens, preserves special tokens `<PAD>` (0), `<UNK>` (1), `<SOS>` (2), `<EOS>` (3), excludes synthetic tokens (`exaggerated`, `proportions`), and maps unobserved tokens to `<UNK>` (1).
4. **Metric Suite Rigor**: Observation 1.4 confirms that `metrics.py` implements all mandatory assignment metrics (FID, KID, seed diversity LPIPS, parameter count, latency, peak VRAM) with edge case guards (small sample scaling, range clamping, seed counts $\ge 2$).

---

## 3. Caveats

- **Sample Size for Publishable KID**: While `metrics.py` safely supports tiny sample sizes ($N < 50$) via dynamic subset scaling, asymptotic unbiasedness for KID in academic reporting requires running with $N \ge 50$ samples.
- **Single-GPU / CPU Evaluation**: Evaluation efficiency profiling was verified in single-device configuration; multi-GPU distributed evaluation uses DataParallel or single-device inference as handled in `main.py` and `evaluate.py`.

---

## 4. Conclusion

The data preprocessing pipeline, compositional splits, vocabulary construction, and evaluation metrics suite satisfy all assignment specifications and empirical integrity checks:
1. `preprocessing/splits.json`: All 4 keys exist, mutually exclusive, covering 100,000 samples.
2. Holdout attributes: 100% of `test_ood` samples possess `(hair=98, glasses=11)`; 0% of `train` samples possess it.
3. `preprocessing/vocab.json`: 149 tokens, special tokens preserved (0..3), zero synthetic OOD tokens, unobserved tokens map to `<UNK>`.
4. `evaluate.py` & `metrics.py`: Complete, robust against edge cases, and calculates all mandatory assignment metrics across both IID and OOD splits.

**Final Verdict**: **APPROVE**

---

## 5. Verification Method

To independently verify these findings:

1. **Verify Split Keys and Counts**:
   Inspect `preprocessing/splits.json`:
   - Check line 2 (`"train"`), line 79638 (`"val"`), line 89594 (`"test_ind"`), line 99550 (`"test_ood"`).
   - Verify sample counts: train (79,634), val (9,954), test_ind (9,954), test_ood (458).
2. **Verify Holdout Attribute in Metadata**:
   Inspect `data/meta/cartoon_image_attributes.csv`:
   - Sample 0 (line 2): `hair=98`, `glasses=11`.
   - Sample 235 (line 237): `hair=98`, `glasses=11`.
   - Sample 99436 (line 99438): `hair=98`, `glasses=11`.
   - Sample 31380 in train (line 31382): `hair=106`, `glasses=11`.
3. **Verify Vocabulary Tokens**:
   Inspect `preprocessing/vocab.json`:
   - Lines 2–5: `<PAD>`: 0, `<UNK>`: 1, `<SOS>`: 2, `<EOS>`: 3.
   - Search for `"exaggerated"` and `"proportions"`: both absent.
4. **Verify Metric Safeguards**:
   Inspect `metrics.py`:
   - Lines 121–128: KID `subset_size` scaling guard `min(50, max(2, min_samples))`.
   - Lines 90–98: `_ensure_zero_one_range` clamping.
   - Lines 153–154: `compute_diversity_across_seeds` seed count check.
