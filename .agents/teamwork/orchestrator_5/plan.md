# Plan — Avatar Diffusion Final Audit & Full Pipeline Test

## Objective
Perform a comprehensive final audit and full-pipeline verification of the Avatar Diffusion codebase to ensure that:
1. Preprocessing and tokenizer fitting work correctly.
2. Training (`train.py`) executes end-to-end (e.g. 1 epoch or short run).
3. Inference (`inference.py`) completes and outputs valid image(s).
4. Evaluation (`evaluate.py`) completes without `RuntimeError`.
5. Natural language prompts seamlessly integrate across all modules without shape mismatches or vocab collisions.
6. Any bugs or deprecations are fixed, verified by Reviewer, Challenger, and Forensic Auditor.

## Milestones
- **M1: Exploration & Pipeline Diagnostics**
  - Dispatch teamwork_preview_explorer agents to survey the entire codebase: `main.py`, `train.py`, `inference.py`, `evaluate.py`, `preprocessing/caption_generator.py`, tokenizer, models, metrics, configs.
  - Assess current status of recent changes (natural language captions vs tokenizer vs dataset vs models).
  - Identify potential points of failure or crashes before execution.

- **M2: Remediation & Alignment Fixes**
  - If any bugs, incompatibilities, or missing files are found, dispatch teamwork_preview_worker to fix them with zero regressions.
  - Ensure strict type safety and architectural constraints.

- **M3: End-to-End Pipeline Execution & Verification**
  - Worker executes end-to-end run:
    - Preprocessing / tokenization check
    - `python train.py --epochs 1` (or short smoke run)
    - `python inference.py`
    - `python evaluate.py`
  - Verify generated artifacts (saved model weights, images, evaluation outputs).

- **M4: Independent Review, Adversarial Challenge & Forensic Audit**
  - Dispatch teamwork_preview_reviewer agents to review code changes and execution results.
  - Dispatch teamwork_preview_challenger agents to stress-test prompt handling and edge cases.
  - Dispatch teamwork_preview_auditor for forensic integrity check (ensuring authentic training, genuine metrics, no dummy/cheating code).

- **M5: Final Synthesis & Sentinel Completion Report**
  - Synthesize reports and submit completion message to Sentinel.
