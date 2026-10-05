# Plan: Avatar Diffusion Defect Remediation & Verification

## Objective
Remediate defects DEF-01 through DEF-20 (DEF-01/09/13/17 already completed in part) by implementing the exact blueprints in `audit_report.md` Section 10 across all remaining modules:
- Preprocessing: `splitter.py`, `tokenizer.py`
- Models: `transformer.py`, `unet_parts.py`, `unet.py`
- Training/Pipeline: `train.py`, `main.py`
- Evaluation/Inference: `metrics.py`, `evaluate.py`, `inference.py`
Followed by rigorous testing, review, adversarial challenge, and forensic integrity audit.

## Milestones & Work Breakdown

### Milestone 1: Core Preprocessing & Tokenization (DEF-01, DEF-02, DEF-05, DEF-09, DEF-13)
- Target files:
  - `preprocessing/splitter.py`: 4-way partition (train 80%, val 10%, test_ind 10%, test_ood held-out), JSON serialization to `splits_path`, ensure > 0 OOD samples.
  - `preprocessing/tokenizer.py`: Word count starting at 3 so new tokens start at ID 4 preserving `<PAD>: 0, <UNK>: 1, <SOS>: 2, <EOS>: 3`. Consistent `_tokenize` stripping punctuation across `fit()` and `encode()`. Safe integer casting in `load_vocab()`.

### Milestone 2: Transformer & U-Net Dynamic Mask Propagation (DEF-03, DEF-04, DEF-15, DEF-16)
- Target files:
  - `models/transformer.py`: `float("-inf")` mask fill in `MultiHeadAttentionBlock.attention`, safe `nan_to_num` fallback.
  - `models/unet_parts.py`: `SpatialCrossAttention` rank-adaptive mask handling (2D, 3D, 4D), boolean mask casting for `scaled_dot_product_attention`.
  - `models/unet.py`: Add `mask=None` parameter across `Unet.forward` and pass it to all cross-attention stages (`attn_inc`, `attn_down1`, `attn_down2`, `attn_bott1`, `attn_up1`, `attn_up2`).

### Milestone 3: Pipeline Integration, Training & Optimization (DEF-04, DEF-10, DEF-11, DEF-14, DEF-19, DEF-20)
- Target files:
  - `train.py`: Explicit mask propagation `unet(..., mask=mask)`, decoupled gradient clipping for unet and text_encoder, validation loss tracking if needed.
  - `main.py`: Unpack 4-way split `(train, val, test_ind, test_ood)`, decoupled AdamW optimizer with selective weight decay and LinearLR warmup + CosineAnnealingLR, deterministic regex checkpoint resolution.

### Milestone 4: Metrics, Standalone Evaluation & CLI Inference (DEF-06, DEF-07, DEF-08, DEF-12, DEF-18, DEF-20)
- Target files:
  - `metrics.py`: `_ensure_zero_one_range` independent normalization, `AttributeAlignmentEvaluator` probe.
  - `evaluate.py`: Standalone batch-accumulating evaluation pipeline for both In-Distribution and OOD splits, deterministic checkpoint resolution.
  - `inference.py`: Deterministic regex checkpoint loading, CLI arguments (`argparse`), DDIM / fast sampling support, non-blocking execution, sample generation verification.

### Milestone 5: Full Verification, Review, Challenge & Forensic Audit
- Verify all acceptance criteria:
  1. `python inference.py` launches and executes without `FileNotFoundError` or missing `vocab.json` crashes.
  2. Preprocessing pipeline generates 4-way split where OOD test set > 0 samples.
  3. Tokenizer preserves special tokens `<UNK>`, `<SOS>`, `<EOS>` without clobbering.
  4. Codebase passes automated test run (single training epoch/batch and evaluation step) without exceptions.
- Independent Reviewers, Challengers, and Forensic Auditor verification.
