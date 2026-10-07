# Handoff Report: Natural Language Captions and Prompts Implementation

## 1. Executive Summary

In accordance with project specifications and task requirements (R1, R2, R3), the avatar diffusion codebase has been updated to use natural language descriptive prompts instead of raw numerical attribute IDs.
All training captions generated from dataset metadata and inference evaluation prompts now use meaningful semantic descriptors (colors, hairstyles, glasses shapes, facial hair styles). No numerical IDs appear in the generated training captions or inference prompts.

---

## 2. Modified Files & Substance of Changes

### 2.1 `preprocessing/caption_generator.py` (Requirement R1)
- **Natural Language Mappings**: Implemented complete semantic mapping tables covering all dataset metadata variants:
  - `FACE_COLORS` (11 variants, 0–10): Maps numerical skin tone codes to descriptive terms (`"porcelain"`, `"brown"`, `"light"`, `"peach"`, `"olive"`, `"tan"`, `"golden"`, `"bronze"`, `"chestnut"`, `"dark"`, `"pale"`).
  - `HAIR_STYLES` (111 variants, 0–110): Maps all 111 hair variants to descriptive hairstyles (e.g., variant 98 mapped to `"wavy"`).
  - `EYE_COLORS` (5 variants, 0–4): Maps eye color codes to natural colors (`"blue"`, `"green"`, `"brown"`, `"hazel"`, `"dark"`).
  - `GLASSES_STYLES` (12 variants, 0–11): Maps glasses codes to natural styles (e.g., variant 0 mapped to `"round glasses"`, variant 11 mapped to `"no glasses"`).
  - `FACIAL_HAIR_STYLES` (15 variants, 0–14): Maps facial hair codes to descriptive styles (e.g., variant 3 mapped to `"full beard"`, variant 14 mapped to `"no facial hair"`).
  - Supplementary mappings added for `HAIR_COLORS` (10 variants), `FACE_SHAPES` (7 variants), and `GLASSES_COLORS` (7 variants).
- **Default Natural Language Template**: Updated to `"a cartoon avatar with {face_color} skin, {hair} hair, {eye_color} eyes, {glasses}, and {facial_hair}"`.
- **Zero Numerical IDs**: Robust `_extract_val` logic extracts attribute codes and falls back to natural language descriptors; guaranteed absence of numerical digits in generated captions.
- **Sequence Length Safety**: Resulting captions average 13–17 words, staying comfortably within `max_seq_len = 20` to avoid truncation.

### 2.2 `inference.py` (Requirement R2)
- **Default Prompt in `generate()`**: Added default parameter `prompt: str = "a blue cartoon avatar with round eyes and exaggerated proportions"`.
- **`ood_prompts` in `evaluate_ood_combinations()`**: Replaced numerical ID strings (`"avatar with face 1, hair 98, eyes 4, glasses 11, and facial hair 3"`, etc.) with natural language prompts:
  1. `"a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard"` (matching the held-out sample attributes for `hair: 98` and `glasses: 11`)
  2. `"a cartoon avatar with wavy hair and no glasses"` (directly isolating the compositional held-out attribute pair)
  3. `"a blue cartoon avatar with round eyes and exaggerated proportions"` (the canonical project prompt)
- **CLI Default**: Ensured `--prompt` default remains the natural language prompt.

---

## 3. Static Code Verification Record (Constraint R3)

In compliance with constraint R3 ("NO TERMINAL EXECUTION: Do not execute any terminal commands to test the code"), verification was conducted through rigorous static analysis:

1. **Syntax & AST Integrity**:
   - `preprocessing/caption_generator.py`: Checked typing imports (`Dict`, `Any`, `Optional`), class scoping, dictionary keys and values, formatting kwargs, exception handling, and `__repr__`.
   - `inference.py`: Verified method signatures, default parameter order, indentation, list syntax in `ood_prompts`.
2. **Type Safety & Robustness**:
   - Both string and integer metadata keys (e.g., `'98'` vs `98`) are normalized via `_extract_val`.
   - Missing or `None` values trigger fallbacks to valid natural language strings.
   - Any unknown attribute values fall back to descriptive terms (`"natural"`, `"styled"`, `"dark"`, `"no glasses"`, `"no facial hair"`), guaranteeing that numerical IDs never leak into the output.
3. **Punctuation & Tokenizer Compatibility**:
   - All descriptors contain only alphabetic characters and single spaces (no attached commas, hyphens, or special symbols).
   - Generated tokens match `AvatarTokenizer._tokenize` expectations.

---

## 4. Acceptance Criteria Checklist

- [x] `preprocessing/caption_generator.py` is updated and maps metadata to strings without numerical IDs.
- [x] `inference.py` uses only natural language prompts for OOD and defaults.
- [x] The code is syntactically valid and type-safe.
