# Handoff Report: Survey of Data Pipeline & Tokenizer

**Target**: Data Ingestion, Metadata Attributes, Compositional Split, and Tokenization  
**Agent**: Survey Explorer 1 (Data & Tokenizer)  
**Parent Conversation ID**: ee9d5fea-6beb-44a8-80ba-060b8747bee8  
**Working Directory**: `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_1`  
**Report Artifact**: `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_1\report.md`  

---

## 1. Observation

1. **`preprocessing/preprocessing_config.json` (lines 7–15)**:
   ```json
     "ood_blocked_combinations": [
       [
         ["color", "blue"],
         ["proportion", "exaggerated"]
       ]
     ],
     "max_seq_len": 20,
     "vocab_path": "preprocessing/vocab.json"
   ```
   The configuration contains `"color": "blue"` and `"proportion": "exaggerated"`. It lacks the `"splits_path"` key.
2. **`data/meta/cartoon_image_attributes.csv` (lines 1–2)**:
   ```csv
   filename,eye_angle,eye_lashes,eye_lid,chin_length,eyebrow_weight,eyebrow_shape,eyebrow_thickness,face_shape,facial_hair,hair,eye_color,face_color,hair_color,glasses,glasses_color,eye_slant,eyebrow_width,eye_eyebrow_distance
   0/cs11556364481883459966.jpg,2,0,0,1,1,13,0,5,3,98,4,1,2,11,2,1,2,2
   ```
   The header contains 18 visual attribute columns. Neither `"color"` nor `"proportion"` exists in the column names. All attribute values are integer strings (e.g. `hair: 98`, `face_color: 1`, `glasses: 11`).
3. **`preprocessing/splitter.py` (lines 13–22, 35–47)**:
   ```python
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
   ```
   and lines 41–47:
   ```python
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
   `splitter.py` is fully implemented matching Blueprint 1.1, returning 4 partitions and writing `splits.json`.
4. **`main.py` (line 54)**:
   ```python
   splitter = CompositionalSplitter(config)
   train_idx, val_idx, ood_idx = splitter.split(raw_metadata)
   ```
   `main.py` unpacks 3 variables from `splitter.split()`, which returns 4 elements.
5. **`preprocessing/config.py` (lines 14–30)**:
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
   `config.py` does not load `splits_path`.
6. **`preprocessing/tokenizer.py` (lines 8–33, 40)**:
   ```python
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
   ```
   `tokenizer.py` already matches Blueprint 1.2, preserving special tokens and sanitizing punctuation in both `fit()` and `encode()`.
7. **Filesystem checks**:
   `preprocessing/vocab.json` and `preprocessing/splits.json` currently do not exist on disk.

---

## 2. Logic Chain

1. **Why 0 OOD samples were generated (`DEF-01`)**:
   - Observation 1 shows `ood_blocked_combinations` contains `{('color', 'blue'), ('proportion', 'exaggerated')}`.
   - Observation 2 shows `cartoon_image_attributes.csv` keys are facial features (`hair`, `glasses`, etc.) and values are integer strings.
   - Observation 3 shows `splitter.py` checks `blocked_set.issubset(current_comb)`.
   - Because `('color', 'blue')` is not in `current_comb` for any row, the condition evaluates to `False` for all rows.
   - Hence, `ood_indices` has length 0 and all rows are assigned to `train_indices`.
   - Modifying `preprocessing_config.json` to `[[["hair", "98"], ["glasses", "11"]]]` matches existing dataset rows (e.g. line 2 in CSV has `hair: 98` and `glasses: 11`), guaranteeing `len(ood_indices) > 0`.
2. **Downstream Unpack Crash in `main.py`**:
   - Observation 3 shows `CompositionalSplitter.split()` returns 4 items: `(final_train_indices, val_indices, test_ind_indices, ood_indices)`.
   - Observation 4 shows `main.py:54` unpacks into 3 variables (`train_idx, val_idx, ood_idx`).
   - Running `main.py` results in a fatal `ValueError: too many values to unpack (expected 3, got 4)`.
   - Therefore, `main.py:54` must be modified to unpack all 4 partitions: `train_idx, val_idx, test_ind_idx, ood_idx`.
3. **Tokenizer Special Token ID Preservation (`DEF-02`)**:
   - Observation 6 shows `word_count` starts at `max(self.vocab.values())` (3).
   - The first added word gets ID 4.
   - Special tokens `<PAD>: 0`, `<UNK>: 1`, `<SOS>: 2`, `<EOS>: 3` are not overwritten.
   - `inverse_vocab` correctly maps 1 to `<UNK>`.
4. **Punctuation Synchronization (`DEF-05`)**:
   - Observation 6 shows `_tokenize` is called by both `fit()` and `encode()`.
   - `re.sub(r'[^\w\s]', '', text.lower())` removes commas before splitting into tokens.
   - Thus, `'1,'` in captions becomes `'1'`, matching integer strings in prompts.
5. **Config Splits Path Parsing**:
   - Observation 5 shows `config.py` omits `self.splits_path = raw_config.get("splits_path", "preprocessing/splits.json")`.
   - While `splitter.py` uses `getattr(self.config, "splits_path", ...)`, omitting it violates Blueprint 1.1 and causes `AttributeError` if accessed directly elsewhere.

---

## 3. Caveats

1. **Model Training & File Generation**:
   `preprocessing/vocab.json` and `preprocessing/splits.json` do not currently exist on disk. They will be generated when `splitter.split()` and `tokenizer.fit(); tokenizer.save_vocab()` run during training or data preparation.
2. **Scope Boundary**:
   As an Explorer, no source code files (`preprocessing_config.json`, `config.py`, `main.py`) were modified. Concrete recommendations and patch targets are detailed in `report.md`.
3. **Alternative Attribute Pairs**:
   `[["hair", "98"], ["glasses", "11"]]` is the exact pair specified in Blueprint 1.1 and row 2 of the CSV. Other valid pairs exist (e.g. `[["hair", "55"], ["glasses", "11"]]`), but adhering to Blueprint 1.1 ensures consistency with default evaluation scripts.

---

## 4. Conclusion

1. **`DEF-01`**: Root cause confirmed. Fix requires updating `preprocessing/preprocessing_config.json` to `[[["hair", "98"], ["glasses", "11"]]]` and adding `"splits_path": "preprocessing/splits.json"`.
2. **`DEF-02` & `DEF-05`**: Fully resolved in `preprocessing/tokenizer.py` (matches Blueprint 1.2).
3. **`DEF-09` & `DEF-13`**: Implemented in `preprocessing/splitter.py` (matches Blueprint 1.1). However, `config.py` needs `self.splits_path`, and `main.py:54` must be updated to unpack 4 variables instead of 3 to prevent an immediate runtime crash.

---

## 5. Verification Method

To verify these findings independently:

1. **Inspect `preprocessing/preprocessing_config.json`**:
   Check lines 7–12 to confirm `["color", "blue"], ["proportion", "exaggerated"]` is present.
2. **Inspect CSV for attribute match**:
   Inspect line 2 of `data/meta/cartoon_image_attributes.csv` to confirm row contains `hair: 98` and `glasses: 11`.
3. **Test Unpack in `main.py`**:
   Inspect line 54 of `main.py` vs line 47 of `preprocessing/splitter.py`. Note that `splitter.split()` returns 4 items while `main.py` unpacks 3 items.
4. **Test Tokenizer**:
   Verify `preprocessing/tokenizer.py` line 25: `word_count = max(self.vocab.values())`.
