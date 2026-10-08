## 2026-10-08T13:45:41Z
You are the SWE Light orchestrator (teamwork_preview_swe).

Your working directory is:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\swe_3

The project root is:
c:\Users\Admin\Desktop\avatar diffusion

The authoritative user request is recorded in:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md
Specifically, see the latest section `## 2026-10-08T13:43:56Z`:

This is a single self-contained modification; keep it small and focused. 
Modify the existing avatar diffusion model to use v-prediction (velocity target) instead of epsilon-prediction. 

Working directory: c:\Users\Admin\Desktop\avatar diffusion
Integrity mode: demo

## Requirements

### R1. Update Training Target
In `train.py`, modify the loss calculation to use the velocity target (`v_target`) instead of the noise target. The velocity target is defined as `sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * original`.

### R2. Update Reverse Process
In `models/diffusion.py`, update the `DiffusionReverseProcess.sample` method to assume the model outputs velocity (`v`). Reconstruct both `pred_x0` and the noise from the predicted velocity for the reverse step.

### R3. Update DDIM Sampling
In `evaluate.py` (`sample_batch`) and `inference.py` (`generate`), update the sampling loops to correctly reconstruct `pred_x0` and the noise from the model's velocity output.

## Acceptance Criteria

### Execution & Verification
- [ ] The agent team must run a quick test using `python train.py --max_steps 2` and confirm it completes without shape mismatches or crashes.
- [ ] The agent team must run a quick test using `python inference.py --num_steps 2` (or similar minimal config) and confirm it generates an output tensor/image without crashing.
- [ ] The mathematical formulas for extracting `x0` and `epsilon` from `v` must mathematically match the definition of v-parameterization.

Please manage the SWE Light workflow: run one implementer on the whole task, then adversarial reviewer rounds carrying a cumulative open-issues ledger, establishing correctness through running tests.
Maintain your `progress.md` in your working directory (`c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\swe_3\progress.md`) so the sentinel can monitor your progress.
When complete, send a message back with your final report and victory claim.

