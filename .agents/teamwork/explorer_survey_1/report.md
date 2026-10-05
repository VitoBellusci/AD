# Comprehensive Forensic Report: Data Pipeline & Tokenization Survey

**Target System**: Tiny Text-Conditioned Avatar Diffusion (`preprocessing/` & dataset metadata)  
**Investigator**: Survey Explorer 1 (Data & Tokenizer)  
**Date**: October 5, 2026  
**Status**: Authoritative Forensic Investigation (Read-Only)  
**Target Files Inspected**:
- `preprocessing/preprocessing_config.json`
- `preprocessing/config.py`
- `preprocessing/splitter.py`
- `preprocessing/tokenizer.py`
- `preprocessing/caption_generator.py`
- `preprocessing/dataset.py`
- `data/meta/cartoon_image_attributes.csv`
- `data/meta/cartoon_attributes_variants.csv`
- `audit_report.md` (Sections 1, 6, 9, 10: Blueprints 1.1 & 1.2; DEF-01, DEF-02, DEF-05, DEF-09, DEF-13)
- `main.py`, `inference.py`, `train.py`

---

## 1. Executive Summary

A comprehensive forensic audit was conducted on the data ingestion, attribute metadata processing, split partitioning, and tokenization subsystems of the Avatar Diffusion repository. The audit evaluated current implementations against the 20 defect catalog (`audit_report.md`), specifically targeting `DEF-01` (0 OOD samples), `DEF-02` (Tokenizer index collisions), `DEF-05` (Caption template punctuation desynchronization), `DEF-09` (Missing ordinary test split), and `DEF-13` (Absence of split persistence).

### Key Findings:
1. **Root Cause of 0 OOD Samples (`DEF-01`) Confirmed**:
   `preprocessing/preprocessing_config.json` lines 7–12 configure blocked attributes as `[["color", "blue"], ["proportion", "exaggerated"]]`. In `data/meta/cartoon_image_attributes.csv`, neither the column names `"color"` / `"proportion"` nor the text string values `"blue"` / `"exaggerated"` exist. The CSV strictly contains 18 integer-coded facial feature categories (`eye_angle`, `hair`, `glasses`, `face_color`, etc.). Consequently, `blocked_set.issubset(current_comb)` evaluates to `False` for 100% of the dataset, producing exactly **0 OOD test samples** and causing the model to train on all data.
2. **State of Blueprint 1.1**:
   - `preprocessing/splitter.py` **already contains** the updated 4-way split logic returning `(final_train_indices, val_indices, test_ind_indices, ood_indices)` and saving to `splits.json`.
   - However, `preprocessing/preprocessing_config.json` has **NOT** been updated (still has `"color": "blue"` and is missing `"splits_path"`).
   - `preprocessing/config.py` has **NOT** been updated (missing `self.splits_path = raw_config.get("splits_path", ...)`).
   - **Critical Unpack Crash in `main.py:54`**: `main.py` still executes `train_idx, val_idx, ood_idx = splitter.split(raw_metadata)`, expecting 3 values while `splitter.split()` returns 4. Running `main.py` currently crashes immediately with `ValueError: too many values to unpack (expected 3, got 4)`.
3. **State of Blueprint 1.2 (`DEF-02`, `DEF-05`)**:
   - `preprocessing/tokenizer.py` **already contains** the Blueprint 1.2 code. `word_count` starts at `max(self.vocab.values())` (starting additions at ID 4, preserving special tokens `<PAD>: 0`, `<UNK>: 1`, `<SOS>: 2`, `<EOS>: 3`). Both `fit()` and `encode()` route through `self._tokenize()` using `re.sub(r'[^\w\s]', '', text.lower()).split()`, which strips punctuation and guarantees complete synchronization.
   - However, no `vocab.json` file currently exists on disk.
4. **Caption & Prompt Semantic Alignment (`DEF-05`)**:
   `CaptionGenerator` constructs prompts using integer attribute codes (`"avatar with face 1, hair 98, eyes 4, glasses 11, and facial hair 3"`). Natural language prompts in `inference.py` lines 161–162 (`"a blue cartoon avatar with round eyes and exaggerated proportions"`) map 100% of tokens to `<UNK>` because natural words do not exist in the training vocabulary.

---

## 2. Component-by-Component Forensic Audit

### 2.1 `preprocessing/preprocessing_config.json`
- **Location**: `preprocessing/preprocessing_config.json:1-16`
- **Current Content**:
  ```json
  {
    "resolution": [64, 64],
    "norm_min": -1.0,
    "norm_max": 1.0,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "ood_blocked_combinations": [
      [
        ["color", "blue"],
        ["proportion", "exaggerated"]
      ]
    ],
    "max_seq_len": 20,
    "vocab_path": "preprocessing/vocab.json"
  }
  ```
- **Defects Detected**:
  - `ood_blocked_combinations` references nonexistent keys and values (`color: blue`, `proportion: exaggerated`).
  - Missing `"splits_path": "preprocessing/splits.json"`.

### 2.2 `preprocessing/config.py`
- **Location**: `preprocessing/config.py:1-37`
- **Current Logic**:
  ```python
  self.resolution: Tuple[int, int] = tuple(raw_config["resolution"])
  self.norm_min: float = raw_config["norm_min"]
  self.norm_max: float = raw_config["norm_max"]
  self.mean: List[float] = raw_config["mean"]
  self.std: List[float] = raw_config["std"]
  self.max_seq_len: int = raw_config["max_seq_len"]
  self.vocab_path: str = raw_config["vocab_path"]
  
  self.ood_blocked_combinations: List[Set[Tuple[str, str]]] = []
  for combination in raw_config.get("ood_blocked_combinations", []):
      rebuilt_set = set(tuple(attr) for attr in combination)
      self.ood_blocked_combinations.append(rebuilt_set)
  ```
- **Defects Detected**:
  - Missing `self.splits_path: str = raw_config.get("splits_path", "preprocessing/splits.json")`.
  - Stale comment referencing `[[["color", "blue"], ["proportion", "exaggerated"]]]`.

### 2.3 `preprocessing/splitter.py`
- **Location**: `preprocessing/splitter.py:1-48`
- **Current Logic**:
  ```python
  class CompositionalSplitter:
      def __init__(self, config):
          self.config = config

      def split(self, metadata_list: List[Dict]) -> Tuple[List[int], List[int], List[int], List[int]]:
          train_indices, ood_indices = [], []
          
          for idx, meta in enumerate(metadata_list):
              current_comb = {(k, str(v)) for k, v in meta.items()}
              is_ood = any(
                  blocked_set.issubset(current_comb) 
                  for blocked_set in self.config.ood_blocked_combinations
              )
              if is_ood:
                  ood_indices.append(idx)
              else:
                  train_indices.append(idx)

          random.seed(42)
          random.shuffle(train_indices)
          n_total = len(train_indices)
          val_size = int(n_total * 0.1)
          test_size = int(n_total * 0.1)
          
          val_indices = train_indices[:val_size]
          test_ind_indices = train_indices[val_size:val_size + test_size]
          final_train_indices = train_indices[val_size + test_size:]

          splits = {
              "train": final_train_indices,
              "val": val_indices,
              "test_ind": test_ind_indices,
              "test_ood": ood_indices
          }
          splits_path = getattr(self.config, "splits_path", "preprocessing/splits.json")
          os.makedirs(os.path.dirname(splits_path) or ".", exist_ok=True)
          with open(splits_path, "w") as f:
              json.dump(splits, f, indent=2)

          return final_train_indices, val_indices, test_ind_indices, ood_indices
  ```
- **Status**:
  - Implementation matches Blueprint 1.1 line-for-line.
  - Correctly implements 4-way partitioning (80% train, 10% val, 10% test_ind, held-out test_ood).
  - Uses `getattr(self.config, "splits_path", "preprocessing/splits.json")` as a fallback, which protects against crashes even if `config.splits_path` is missing.
  - Saves to `splits.json`.

### 2.4 `preprocessing/tokenizer.py`
- **Location**: `preprocessing/tokenizer.py:1-64`
- **Current Logic**:
  ```python
  class AvatarTokenizer:
      def __init__(self, config=None):
          self.config = config
          self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
          self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

      def _tokenize(self, text: str) -> List[str]:
          clean_text = re.sub(r'[^\w\s]', '', text.lower())
          return clean_text.split()

      def fit(self, training_texts: List[str]):
          word_count = max(self.vocab.values()) # Inizia da 3 -> il primo token aggiunto avrà ID 4
          for text in training_texts:
              tokens = self._tokenize(text)
              for token in tokens:
                  if token not in self.vocab:
                      word_count += 1
                      self.vocab[token] = word_count
          self.inverse_vocab = {v: k for k, v in self.vocab.items()}

      def encode(self, text: str) -> List[int]:
          tokens = self._tokenize(text)
          encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
          max_seq_len = self.config.max_seq_len if self.config else 20
          if len(encoded) < max_seq_len:
              encoded += [self.vocab["<PAD>"]] * (max_seq_len - len(encoded))
          else:
              encoded = encoded[:max_seq_len]
          return encoded
  ```
- **Status**:
  - Matches Blueprint 1.2 line-for-line.
  - Special tokens are safely preserved at 0, 1, 2, 3.
  - Vocabulary counter begins at 3 (`word_count = max(self.vocab.values())`), so the first new word receives ID 4.
  - Punctuation stripping is uniformly shared between `fit()` and `encode()`.

### 2.5 `preprocessing/caption_generator.py`
- **Location**: `preprocessing/caption_generator.py:1-30`
- **Current Logic**:
  ```python
  self.template = (
      "avatar with face {face_color}, hair {hair}, "
      "eyes {eye_color}, glasses {glasses}, and facial hair {facial_hair}"
  )
  ```
- **Analysis**:
  - Generates deterministic strings based on 5 prominent visual attributes from the CSV: `face_color`, `hair`, `eye_color`, `glasses`, `facial_hair`.
  - Trailing commas are removed during `_tokenize()` by regex `re.sub(r'[^\w\s]', '', text.lower())`, so `'1,'` becomes `'1'`.

### 2.6 `preprocessing/dataset.py`
- **Location**: `preprocessing/dataset.py:1-54`
- **Current Logic**:
  - Transforms: `transforms.Resize(self.config.resolution)`, `transforms.ToTensor()`, `transforms.Normalize(mean=self.config.mean, std=self.config.std)`.
  - Maps PIL images to normalized float tensors in $[-1.0, 1.0]$.
  - Invokes `self.tokenizer.encode(caption_text)` and returns `(image_tensor, caption_tensor)`.

### 2.7 Metadata Files: `data/meta/cartoon_image_attributes.csv` & `variants.csv`
- Header of `cartoon_image_attributes.csv`:
  `filename,eye_angle,eye_lashes,eye_lid,chin_length,eyebrow_weight,eyebrow_shape,eyebrow_thickness,face_shape,facial_hair,hair,eye_color,face_color,hair_color,glasses,glasses_color,eye_slant,eyebrow_width,eye_eyebrow_distance`
- Total samples: 100,000 images (lines 2 to 100,001 in CSV).
- Variants in `cartoon_attributes_variants.csv`:
  - `hair`: 111 variants (0 to 110)
  - `glasses`: 12 variants (0 to 11)
  - `face_color`: 11 variants (0 to 10)
  - `facial_hair`: 15 variants (0 to 14)
  - `eye_color`: 5 variants (0 to 4)
- Verification of target OOD pair:
  Row 2 (`0/cs11556364481883459966.jpg`) contains:
  `hair = 98`, `glasses = 11`, `face_color = 1`, `eye_color = 4`, `facial_hair = 3`.
  Targeting `[["hair", "98"], ["glasses", "11"]]` guarantees a valid, non-empty OOD set.

---

## 3. Deep-Dive: Root Cause of 0 OOD Samples (`DEF-01`)

### 3.1 Trace of Failure Mechanism
1. In `splitter.py`:
   ```python
   current_comb = {(k, str(v)) for k, v in meta.items()}
   is_ood = any(
       blocked_set.issubset(current_comb) 
       for blocked_set in self.config.ood_blocked_combinations
   )
   ```
2. When loaded from current `preprocessing_config.json`:
   `self.config.ood_blocked_combinations` contains `{('color', 'blue'), ('proportion', 'exaggerated')}`.
3. Every dictionary in `metadata_list` contains the following keys:
   `{'eye_angle', 'eye_lashes', 'eye_lid', 'chin_length', 'eyebrow_weight', 'eyebrow_shape', 'eyebrow_thickness', 'face_shape', 'facial_hair', 'hair', 'eye_color', 'face_color', 'hair_color', 'glasses', 'glasses_color', 'eye_slant', 'eyebrow_width', 'eye_eyebrow_distance'}`.
4. Neither key `'color'` nor `'proportion'` exists in `current_comb`.
5. Therefore, `blocked_set.issubset(current_comb)` returns `False` unconditionally:
   $$\forall \text{sample} \in \mathcal{D}, \quad \text{is\_ood}(\text{sample}) = \text{False}$$
6. As a result:
   - `len(ood_indices) == 0`.
   - `len(train_indices) == 100,000`.
   - The model is trained on 100% of all attribute combinations.
   - The core research inquiry of the academic assignment ("Can the tiny diffusion model generalize to unseen attribute combinations?") is completely unanswerable because no held-out combinations exist.

### 3.2 Mathematical Verification of the Blueprint 1.1 Fix
Under Blueprint 1.1:
`ood_blocked_combinations` is configured as:
`[["hair", "98"], ["glasses", "11"]]`.
In `splitter.py`:
`blocked_set` becomes `{('hair', '98'), ('glasses', '11')}`.
Assuming approximately uniform distribution of attributes across 100,000 avatars:
$$\mathbb{E}[N_{\text{OOD}}] = \frac{100{,}000}{|\text{variants}(\text{hair})| \times |\text{variants}(\text{glasses})|} = \frac{100{,}000}{111 \times 12} \approx 75 \text{ samples}$$
Row 2 (`0/cs11556364481883459966.jpg`) is confirmed to have `hair = 98` and `glasses = 11`.
Thus, $N_{\text{OOD}} > 0$, creating a controlled, held-out compositional test set.

---

## 4. Deep-Dive: Tokenizer Index Collision & Special Token Overwriting (`DEF-02`)

### 4.1 Trace of Bug in Original Implementation
In original `AvatarTokenizer.fit()`:
```python
def fit(self, training_texts: List[str]):
    word_count = 0  # <--- CRITICAL ERROR
    for text in training_texts:
        tokens = text.lower().split()
        for token in tokens:
            if token not in self.vocab:
                word_count += 1
                self.vocab[token] = word_count
    self.inverse_vocab = {v: k for k, v in self.vocab.items()}
```

Initial vocabulary:
`{"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}`

Collision sequence:
| Step | Token Processed | `word_count` | `self.vocab` Action | Collision With | Inverse Vocab Overwrite |
|---|---|---|---|---|---|
| Init | `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>` | 0 | Pre-populated | None | `{0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}` |
| Word 1 | `'avatar'` | 1 | `vocab['avatar'] = 1` | `<UNK>` (ID 1) | `inverse_vocab[1]` becomes `'avatar'` |
| Word 2 | `'with'` | 2 | `vocab['with'] = 2` | `<SOS>` (ID 2) | `inverse_vocab[2]` becomes `'with'` |
| Word 3 | `'face'` | 3 | `vocab['face'] = 3` | `<EOS>` (ID 3) | `inverse_vocab[3]` becomes `'face'` |
| Word 4 | `'1'` | 4 | `vocab['1'] = 4` | None (safe) | `inverse_vocab[4] = '1'` |

### 4.2 Impact of the Bug
1. Special tokens `<UNK>`, `<SOS>`, and `<EOS>` are overwritten in `self.inverse_vocab`.
2. Any out-of-vocabulary token encountered during inference maps to `self.vocab["<UNK>"] = 1`.
3. In `FullTextEncoder`, index 1 maps to the embedding vector of `'avatar'` rather than an independent unknown token representation.
4. When decoded with `decode([1])`, it prints `'avatar'` instead of `'<UNK>'`.

### 4.3 Blueprint 1.2 Verification
Blueprint 1.2 modifies the initialization:
```python
word_count = max(self.vocab.values()) # Evaluates to 3
```
- First new word receives $3 + 1 = 4$.
- Special tokens 0, 1, 2, 3 remain intact and completely isolated.
- Verified in current `preprocessing/tokenizer.py:25`.

---

## 5. Deep-Dive: Punctuation Desynchronization & Caption Semantics (`DEF-05`)

### 5.1 Trailing Comma Problem
In `CaptionGenerator`:
`"avatar with face {face_color}, hair {hair}, eyes {eye_color}, glasses {glasses}, and facial hair {facial_hair}"`
Formatting row 2 yields:
`"avatar with face 1, hair 98, eyes 4, glasses 11, and facial hair 3"`

Naive whitespace splitting (`text.split()`) produces tokens:
`'avatar'`, `'with'`, `'face'`, `'1,'`, `'hair'`, `'98,'`, `'eyes'`, `'4,'`, `'glasses'`, `'11,'`, `'and'`, `'facial'`, `'hair'`, `'3'`

If a user at inference prompts:
`"avatar with face 1 hair 98 eyes 4 glasses 11 and facial hair 3"`
The tokens extracted are `'1'`, `'98'`, `'4'`, `'11'`.
Because the vocabulary contains `'1,'` rather than `'1'`, every attribute maps to `<UNK>` (ID 1).

### 5.2 Synchronization Mechanism
In `preprocessing/tokenizer.py`, `_tokenize` is applied symmetrically:
```python
def _tokenize(self, text: str) -> List[str]:
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    return clean_text.split()
```
- Strips `,`, `.`, and all punctuation before splitting.
- Both `"face 1,"` and `"face 1"` resolve to `['face', '1']`.
- Ensures identical token mapping during training `fit()` and inference `encode()`.

### 5.3 Natural Language Prompt Disconnect
In `inference.py` lines 161–162:
```python
ood_test_prompts = [
    "a blue cartoon avatar with round eyes and exaggerated proportions",
    "a red avatar with standard eyes and normal proportions"
]
```
Words `'blue'`, `'cartoon'`, `'round'`, `'exaggerated'`, `'proportions'`, `'red'`, `'standard'`, `'normal'` are never present in the training set (which only uses the fixed template with integer attribute values).
Consequently, `[self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]` evaluates to:
`[1, 1, 1, 4, 1, 1, 1, 1, 1]` — virtually all `<UNK>`!
**Recommendation**: The default prompts and OOD test prompts in `inference.py` must use the attribute vocabulary (e.g. `"avatar with face 1, hair 98, eyes 4, glasses 11, and facial hair 3"`) as specified in Blueprint 3.3.

---

## 6. Deep-Dive: 4-Way Dataset Split & Persistence (`DEF-09`, `DEF-13`)

### 6.1 Split Ratios & Partition Definitions
`CompositionalSplitter.split()` partitions the dataset as follows:
1. Identify all OOD samples: $\mathcal{D}_{\text{OOD}} = \{ x \in \mathcal{D} \mid \text{hair}(x) = 98 \land \text{glasses}(x) = 11 \}$.
2. In-distribution pool: $\mathcal{D}_{\text{In}} = \mathcal{D} \setminus \mathcal{D}_{\text{OOD}}$.
3. Shuffle $\mathcal{D}_{\text{In}}$ with fixed seed 42.
4. Partitions:
   - **Validation Set** (`val`): $10\%$ of $\mathcal{D}_{\text{In}}$ ($\approx 9{,}990$ samples).
   - **Ordinary Test Set** (`test_ind`): $10\%$ of $\mathcal{D}_{\text{In}}$ ($\approx 9{,}990$ samples).
   - **Training Set** (`train`): $80\%$ of $\mathcal{D}_{\text{In}}$ ($\approx 79{,}940$ samples).
   - **OOD Test Set** (`test_ood`): $100\%$ of $\mathcal{D}_{\text{OOD}}$ ($\approx 75$ samples).

### 6.2 Serialization & Persistence (`splits.json`)
The partitions are saved to `splits_path` (default `preprocessing/splits.json`):
```json
{
  "train": [ ... ],
  "val": [ ... ],
  "test_ind": [ ... ],
  "test_ood": [ ... ]
}
```
This guarantees exact split reproducibility for downstream evaluation in `evaluate.py`.

### 6.3 Downstream Conflict in `main.py:54`
In `splitter.py:47`:
```python
return final_train_indices, val_indices, test_ind_indices, ood_indices
```
In `main.py:54`:
```python
train_idx, val_idx, ood_idx = splitter.split(raw_metadata)
```
- Expected unpacked items: 3.
- Actual returned items: 4.
- **Immediate Failure**: `ValueError: too many values to unpack (expected 3, got 4)`.
- **Fix Required in `main.py`**:
  ```python
  train_idx, val_idx, test_ind_idx, ood_idx = splitter.split(raw_metadata)
  ```

---

## 7. Line-by-Line Blueprint Comparison (Blueprints 1.1 & 1.2 vs. Codebase)

### 7.1 Blueprint 1.1 Comparison

| Blueprint 1.1 Component | Audit Report Spec (Section 10.1) | Current Codebase Status | Discrepancy / Action Needed |
|---|---|---|---|
| `preprocessing_config.json` | `"splits_path": "preprocessing/splits.json"` | **Missing** | Add `"splits_path": "preprocessing/splits.json"` |
| `preprocessing_config.json` | `"ood_blocked_combinations": [[["hair", "98"], ["glasses", "11"]]]` | **Non-compliant** (has `[["color", "blue"], ["proportion", "exaggerated"]]`) | Replace with `[[["hair", "98"], ["glasses", "11"]]]` |
| `config.py: PreprocessingConfig.__init__` | `self.splits_path: str = raw_config.get("splits_path", "preprocessing/splits.json")` | **Missing** | Add `self.splits_path` parsing to `config.py` |
| `splitter.py: CompositionalSplitter.split` | Generates 4-way split (`train`, `val`, `test_ind`, `test_ood`) | **Fully Compliant** | Already applied |
| `splitter.py: Persistence` | Writes `splits.json` to disk | **Fully Compliant** | Already applied |

### 7.2 Blueprint 1.2 Comparison

| Blueprint 1.2 Component | Audit Report Spec (Section 10.1) | Current Codebase Status | Discrepancy / Action Needed |
|---|---|---|---|
| `tokenizer.py: _tokenize` | `re.sub(r'[^\w\s]', '', text.lower()).split()` | **Fully Compliant** | Already applied |
| `tokenizer.py: fit` | `word_count = max(self.vocab.values())` (starts at ID 4) | **Fully Compliant** | Already applied |
| `tokenizer.py: encode` | Routes through `self._tokenize(text)` | **Fully Compliant** | Already applied |
| `tokenizer.py: decode` | Joins via `inverse_vocab` | **Fully Compliant** | Already applied |
| `tokenizer.py: save_vocab / load_vocab` | Serializes and parses dictionary | **Fully Compliant** | Already applied |

---

## 8. Summary of Defect Resolutions for Implementation Team

1. **`preprocessing/preprocessing_config.json`**:
   Update `ood_blocked_combinations` to `[[["hair", "98"], ["glasses", "11"]]]` and add `"splits_path": "preprocessing/splits.json"`.
2. **`preprocessing/config.py`**:
   Add `self.splits_path: str = raw_config.get("splits_path", "preprocessing/splits.json")` to `PreprocessingConfig.__init__`.
3. **`main.py:54`**:
   Update line 54 to unpack all 4 partitions: `train_idx, val_idx, test_ind_idx, ood_idx = splitter.split(raw_metadata)`.
4. **`inference.py`**:
   Ensure `vocab.json` exists before calling `load_vocab()`, and update test prompts from natural language words to valid attribute template prompts matching `CaptionGenerator`.
