# Handoff Report: Specification Mining for Text-Conditioned Avatar Diffusion

**Agent:** `spec_miner_survey_1`  
**Milestone:** Specification Mining & Requirements Cataloging  
**Target File:** `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\spec_miner_survey_1\spec_requirements.md`  
**Date:** 2026-10-05  

---

## 1. Observation

Direct examination of the authoritative academic assignment specification `Deep_Learning_2026_VI 1.pdf` and codebase files yielded the following primary evidence:

1. **Academic Context (`Deep_Learning_2026_VI 1.pdf`, Page 1):**
   - Header: "Deep Learning - Instructor: Vito Walter Anelli, Ph.D. / M.D. in Computer Engineering - Politecnico di Bari / Test code: 2026_VI - Released on September 21st, 2026 / Deadline: October 22nd, 2026".
   - Central Research Question: *"Can a very small diffusion model, whose denoiser and text encoder are both trained from scratch, generalize to combinations of avatar attributes that were not observed together during training?"*
   - Core Target: "complete pixel-space DDPM pipeline... generates cartoon avatars from short structured descriptions... 32x32 or 64x64 image."

2. **Mandatory From-Scratch Constraints (`Deep_Learning_2026_VI 1.pdf`, Pages 2-3):**
   - Explicitly forbidden components:
     - "Stable Diffusion, Tiny-SD, SDXL, Flux, or any other pretrained diffusion checkpoint;"
     - "CLIP, T5, BERT, or any pretrained text encoder used for conditioning;"
     - "a pretrained VAE or latent diffusion pipeline;"
     - "a ready-made Diffusers training pipeline used as a black box;"
     - "pretrained image or text embeddings as the main representation."
   - Permitted primitives: "Standard layers such as convolutions, normalization, embeddings, and attention are allowed, but the U-Net forward pass, text encoder, conditioning path, diffusion objective, noise schedule, and reverse sampling loop must be implemented and documented by you."

3. **Architectural Envelope (`Deep_Learning_2026_VI 1.pdf`, Page 3):**
   - "base channels: 64-128;"
   - "two or three spatial resolutions;"
   - "residual blocks with time embeddings;"
   - "a 2-4 layer Transformer text encoder trained from scratch;"
   - "text hidden size: 64-128;"
   - "conditioning injected through cross-attention or other techniques."

4. **Data and Compositional Split Requirements (`Deep_Learning_2026_VI 1.pdf`, Pages 2-3):**
   - Dataset: Google Cartoon Set (10k fallback, 100k extended).
   - Captions: "converted into deterministic, multi-attribute captions. For example: 'a boy with blue colors, round eyes, and exaggerated proportions'."
   - Split rules: "The main split must be defined over attribute combinations rather than only over individual images. Hold out a controlled set of combinations... You must report: which individual attributes appear in training; which combinations are held out; the number of train, validation, ordinary test, and compositional-OOD examples; how the caption vocabulary is constructed without using validation or test text."

5. **Evaluation Metrics and Baselines (`Deep_Learning_2026_VI 1.pdf`, Pages 3-4):**
   - Minimum required experiments:
     - "unconditional neural baseline;"
     - "conditional model with the selected conditioning mechanism;"
   - Mandatory metrics:
     - "image quality metric such as FID or KID, with all implementation details documented;"
     - "diversity across seeds for the same prompt;"
     - "parameter count, sampling time, and memory usage."

6. **Current Implementation Observations in Codebase:**
   - In `preprocessing/splitter.py` (lines 47–52): The splitter only returns `train_indices, val_indices, ood_indices`, omitting the required 4th partition: `ordinary test`.
   - In `preprocessing/preprocessing_config.json` (lines 7–12): `"ood_blocked_combinations": [[["color", "blue"], ["proportion", "exaggerated"]]]`, whereas the dataset metadata (`data/meta/cartoon_attributes_variants.csv`) uses keys such as `face_color`, `hair`, `eye_color`, `glasses`, and `facial_hair`.
   - In `models/unet.py` and `models/diffusion.py`: A from-scratch U-Net and DDPM Cosine Scheduler are implemented with CFG and FlashAttention.

---

## 2. Logic Chain

1. **Step 1 (Source Verification):** Comparison between `ORIGINAL_REQUEST.md` and `Deep_Learning_2026_VI 1.pdf` establishes that the assignment is an academic project from Politecnico di Bari requiring an uncompromised from-scratch implementation.
2. **Step 2 (Constraint Formulation):** Any presence of pretrained weights or Hugging Face Diffusers pipelines will trigger immediate disqualification under PDF §4. Therefore, the specification must catalogue every component that must be student-authored.
3. **Step 3 (Research Question Analysis):** The central research question requires testing whether the model generalizes to attribute combinations unobserved during training. This dictates that:
   - All individual attributes must be present in training;
   - Specific pairwise or higher-order attribute combinations must be completely blocked from training;
   - An ordinary test set (seen combinations) AND an OOD test set (unseen combinations) must be evaluated and compared;
   - Vocabulary construction must be isolated to avoid leaking test attribute words.
4. **Step 4 (Metric and Baseline Mapping):** Demonstrating the effect of text conditioning requires comparing an unconditional baseline against the conditional model. Evaluating generative fidelity demands FID/KID and diversity (LPIPS across seeds), while resource compliance on the T4 GPU requires tracking parameters, latency, and VRAM.
5. **Step 5 (Exhaustive Synthesis):** These requirements were synthesized into a structured specification document (`spec_requirements.md`) with concrete numerical bounds, mathematical formulations, feature tables, edge case behaviors, and a traceability matrix.

---

## 3. Caveats

- **No Caveats Regarding the Assignment Specification:** The assignment PDF was retrieved and read in its entirety (all 4 pages) with zero omissions.
- **Auditing Boundary:** This agent's role was strictly read-only specification mining. Code compliance auditing and gap analysis will be performed by subsequent audit agents using `spec_requirements.md` as the authoritative benchmark.

---

## 4. Conclusion

All academic constraints, architectural envelopes, data partitioning rules, mathematical DDPM formulas, required baselines, evaluation metrics, and deliverables have been successfully extracted and codified in `spec_requirements.md`. Key findings include:
1. Strict prohibition of pretrained weights (CLIP, Stable Diffusion, T5, BERT, VAE, Diffusers pipelines).
2. Mandated 4-way compositional split (`train`, `val`, `ordinary test`, `compositional OOD`) with strict vocabulary isolation.
3. Compact parameter envelope: base channels 64–128, 2–3 spatial resolutions, 2–4 layer text transformer, $d_{\text{model}} \in [64, 128]$.
4. Required experimental runs: unconditional baseline vs. conditional model, plus multi-seed diversity and 32x32 smoke test.
5. Required quantitative metrics: FID/KID, pairwise LPIPS across seeds, parameter breakdown, sampling time, and peak VRAM.

---

## 5. Verification Method

To independently verify this specification extraction:
1. **Inspect Specification Document:**
   ```powershell
   Get-Content ".agents\teamwork\spec_miner_survey_1\spec_requirements.md"
   ```
2. **Cross-Reference Authoritative Assignment PDF:**
   - Open `Deep_Learning_2026_VI 1.pdf` and check:
     - Page 1: Task Overview & Research Question.
     - Page 2: Mandatory From-Scratch Constraints & Dataset Splitting.
     - Page 3: Model Architecture Envelope & Training Hints.
     - Page 4: Mandatory Metrics & Deliverables.
3. **Invalidation Conditions:**
   - The specification is invalidated if any requirement in `Deep_Learning_2026_VI 1.pdf` is missing from `spec_requirements.md` or if any requirement contradicts the text of the PDF.
