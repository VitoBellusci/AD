# Execution Plan — Avatar Diffusion Final Verification & Fix-up

## Objective
Perform a comprehensive final code review, functional verification, and fix-up of the Avatar Diffusion project. Ensure strict compliance with all assignment requirements, zero bugs, and full readiness for the definitive training run.

## Plan Steps
1. **Phase 0: Survey & Scope Mapping (Exploration)**
   - Dispatch `spec_miner_survey_6_1`: Mine requirements from `pdf_content.txt`, `audit_report.md`, and `ORIGINAL_REQUEST.md`.
   - Dispatch `explorer_survey_6_2`: Explore data preprocessing, dataset splits, deterministic multi-attribute captions, vocabulary construction, and compositional split logic (`preprocessing/`, `data/`).
   - Dispatch `explorer_survey_6_3`: Explore model architectures (`models/` - U-Net, text tokenizer & encoder, cross-attention, DDPM scheduling), training pipeline (`train.py`, `main.py`), inference (`inference.py`), and evaluation metrics (`evaluate.py`, `metrics.py`).
   - Collect findings and merge into `PROJECT.md` at project root with Feature Inventory, Milestones, and Interface Contracts.

2. **Phase 1: Remediation & Alignment Fixes (Worker)**
   - Dispatch Worker(s) to address any defects found:
     - Verify and enforce R1: resizing, consistent normalization, deterministic multi-attribute captions, vocabulary constructed *only* from training split, compositional split strictly holding out specific attribute combination.
     - Verify and enforce R2: from-scratch tokenizer, 2-4 layer Transformer text encoder, pixel-space DDPM U-Net denoiser, no pretrained weights or forbidden libraries.
     - Verify and enforce R3: correct cross-attention conditioning, noise schedule, forward noising, reverse sampling, checkpointing.
     - Verify and enforce R4: evaluation metrics (FID or KID, diversity across seeds, parameter count, sampling time, memory usage) on standard and compositional OOD prompts.
   - Worker runs short verification tests (dummy training epoch, reverse sampling test, metrics test).

3. **Phase 2: Independent Review & Empirical Challenge**
   - Dispatch 2 independent Reviewers (`teamwork_preview_reviewer`) to inspect code compliance, architectural soundness, and interface conformance.
   - Dispatch 2 Challengers (`teamwork_preview_challenger`) to empirically test edge cases, out-of-distribution prompts, tensor shapes, reproducibility, and memory behavior.

4. **Phase 3: Forensic Integrity Audit**
   - Dispatch Forensic Auditor (`teamwork_preview_auditor`) to verify zero pretrained weights, zero hardcoded results, authentic custom modules, and genuine training/sampling logic.
   - Enforce hard binary veto.

5. **Phase 4: Gate Check, Synthesis & Sentinel Report**
   - Record verdicts in `GATE_STATUS.md`.
   - Synthesize all findings and verified outputs.
   - Send final project completion report via `send_message` to parent Sentinel (`cb6bd98e-ca78-40af-9377-a561a9a1ee8e`).
