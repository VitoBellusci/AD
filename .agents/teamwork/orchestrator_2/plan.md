# Plan: Avatar Diffusion Defect Remediation (Section 10 Blueprints)

## Mission & Scope
Remediate defects DEF-01 through DEF-20 (excluding already applied DEF-17 and DEF-19) strictly following Section 10 blueprints from `audit_report.md`.
Ensure zero regressions, strict adherence to from-scratch and Tiny model constraints, and 100% pass rate on acceptance criteria.

## Milestones Overview

### Milestone 1: Preprocessing & Tokenization Remediations
- **Scope**:
  - `preprocessing/preprocessing_config.json`: Update OOD blocked combinations to actual CSV attributes (e.g. `[["hair", "98"], ["glasses", "11"]]`), parse `splits_path`.
  - `preprocessing/config.py`: Add `splits_path` parsing with default, safe attribute extraction.
  - `preprocessing/splitter.py`: Implement 4-way split (Train 80%, Val 10%, Test Ind 10%, OOD held-out), persist splits to `splits_path` (JSON), verify > 0 OOD samples.
  - `preprocessing/tokenizer.py`: Fix index collision (`word_count` starting after special tokens ID 3 -> next ID 4), synchronize `_tokenize` with punctuation stripping across `fit()` and `encode()`.
  - `preprocessing/caption_generator.py`: Verify consistency with tokenization.
- **Defects Addressed**: DEF-01, DEF-02, DEF-05, DEF-09, DEF-13.

### Milestone 2: Architecture & Mask Propagation Remediations
- **Scope**:
  - `models/transformer.py`: Replace `-1e-9` mask with `float("-inf")` in `MultiHeadAttentionBlock.attention`, safe NaN handling, and vector parameter LayerNormalization.
  - `models/unet_parts.py`: Implement rank-adaptive mask handling in `SpatialCrossAttention.forward` supporting 2D, 3D, 4D masks and boolean casting for `scaled_dot_product_attention`.
  - `models/unet.py`: Add `mask=None` parameter across `Unet.forward` and pass it to all cross-attention layers.
  - `models/diffusion.py`: Verify reverse process sample dynamic range clipping and CFG mask concatenation.
- **Defects Addressed**: DEF-03, DEF-04, DEF-15, DEF-16, DEF-17.

### Milestone 3: Training & Checkpoints Remediations
- **Scope**:
  - `train.py`: Wire `mask` into `unet` forward call (`predicted_noise = unet(noisy_images, timesteps, context, mask=mask)`), decoupled gradient clipping, validation loss evaluation step, AMP support if configured.
  - `main.py`: Use 4-way splits, configure decoupled optimizers (AdamW with selective weight decay and LR warmup), support unconditional baseline flag, deterministic checkpoint loading via regex.
- **Defects Addressed**: DEF-10, DEF-11, DEF-14, DEF-19, DEF-20.

### Milestone 4: Evaluation & Usability Suite Remediations
- **Scope**:
  - `metrics.py`: Independent zero-one range normalization (`_ensure_zero_one_range`), attribute alignment evaluator probe (`AttributeAlignmentEvaluator`).
  - `evaluate.py`: Implement standalone batch-accumulating evaluation pipeline evaluating both In-Distribution and OOD splits with deterministic checkpoint resolution.
  - `inference.py`: Deterministic checkpoint resolution via regex, CLI arguments (`argparse`), 50-step DDIM / fast sampling support, non-blocking automation, sample generation verification.
- **Defects Addressed**: DEF-06, DEF-07, DEF-08, DEF-12, DEF-18.

### Milestone 5: Full Zero-Regression Acceptance & Forensic Integrity Audit
- **Scope**:
  - Verify acceptance criteria:
    1. `python inference.py` launches and executes without `FileNotFoundError` or missing `vocab.json` crashes.
    2. Preprocessing generates 4-way split where OOD test set > 0 samples.
    3. Tokenizer preserves `<UNK>`, `<SOS>`, `<EOS>` without clobbering, correctly tokenizes attribute prompts.
    4. Codebase passes automated test run: 1 training epoch / batch and 1 evaluation step without exceptions.
  - Reviewer and Challenger verification.
  - Forensic Auditor integrity verification (CLEAN).
