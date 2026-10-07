## 2026-10-07T14:10:25Z
You are worker_remediation_6_1.
Your working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_remediation_6_1
Your parent is orchestrator_6 (conversation ID: ebf019cd-ccf0-44f1-97b0-e8b532cc9e09).

MANDATORY FIRST STEP: Read the user request history in:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md
Also read the authoritative project document:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_6\PROJECT.md
And the handoff reports from the survey team:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\spec_miner_survey_6_1\handoff.md
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_6_2\handoff.md
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_6_3\handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

EXCLUSIVE WRITE OWNERSHIP:
You own and may edit the following files:
- main.py
- train.py
- evaluate.py
- metrics.py
- inference.py
- models/diffusion.py
- preprocessing/splits.json
- preprocessing/vocab.json

TASKS TO IMPLEMENT:
1. Fix Vocabulary Data Leakage:
   In `main.py`, fit the tokenizer strictly on `train_texts` (do NOT prepend `canonical_prompts`). Save the updated vocabulary to `preprocessing/vocab.json`. Ensure no OOD synthetic words ("exaggerated", "proportions") exist in `vocab.json`.
2. Regenerate `preprocessing/splits.json`:
   Execute the compositional splitting logic using `CompositionalSplitter` so `preprocessing/splits.json` contains all four disjoint partitions: `"train"`, `"val"`, `"test_ind"`, and `"test_ood"`. Verify that `"test_ood"` is non-empty (holds out `hair=98` and `glasses=11`, ~459 samples) and 100% disjoint from `"train"`.
3. Add CLI Runner to `train.py`:
   Add `if __name__ == "__main__": from main import main; main()` to the end of `train.py` so `python train.py --epochs 1` executes seamlessly.
4. Align Default Prompt in `inference.py`:
   Change default prompt in `inference.py:130` and `inference.py:294` from `"a blue cartoon avatar with round eyes and exaggerated proportions"` to a valid natural language prompt from the dataset domain, such as the held-out OOD composition: `"a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard"`.
5. Guard Multi-GPU RNG state in `main.py`:
   In `main.py:230-234`, guard `set_rng_state_all` so that if the saved CUDA RNG states count does not match the current machine's `torch.cuda.device_count()`, it falls back safely without raising `RuntimeError`.
6. Full Evaluation Integration in `evaluate.py` & `metrics.py`:
   - In `evaluate.py`, invoke `evaluator.compute_efficiency_metrics(unet, text_encoder, device)` and log total parameters, U-Net params, text encoder params, sampling latency, and peak VRAM.
   - In `evaluate.py`, invoke `evaluator.compute_diversity_across_seeds` to compute and log pairwise LPIPS diversity across seeds for prompts.
   - In `evaluate.py`, add `--num_steps` argument and DDIM fast sampling support so evaluation can be executed with custom step counts (e.g. 10, 20, 50, or 1000) rather than hardcoding 1000 DDPM steps per batch.
   - In `metrics.py`, guard KID calculation: if `len(features) < 50` or `num_samples < 50`, set `subset_size = min(50, max(2, len(features)))` or safely handle small sample counts to prevent `torchmetrics` crashes.
7. Support Linear Schedule in `models/diffusion.py`:
   In `DiffusionScheduler`, add a `schedule_type="cosine"` (default) or `"linear"` option to support both Ho et al. linear beta schedule ($10^{-4}$ to $0.02$) and Nichol-Dhariwal cosine schedule.

FUNCTIONAL VERIFICATION TO EXECUTE:
After implementing all fixes, execute and verify:
1. Preprocessing & Split verification: Run a python check verifying `splits.json` contains all 4 keys, `test_ood` is non-empty, and `vocab.json` has only train tokens.
2. Short dummy training run: Run `python train.py --epochs 1 --batch_size 16` (or `python main.py --epochs 1 --batch_size 16`) for 1 epoch or a few steps on CPU/GPU. Ensure it completes with exit code 0 and saves a checkpoint.
3. Inference reverse sampling run: Run `python inference.py --num_steps 10 --prompt "a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard" --output_dir outputs`. Ensure it generates an image file without error.
4. Evaluation run: Run `python evaluate.py --num_samples 10 --num_steps 10 --batch_size 5`. Ensure it computes and logs FID/KID, diversity across seeds, parameter counts, sampling time, and peak VRAM without crashing.

OUTPUT:
Deliver your report at:
`c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_remediation_6_1\handoff.md`
Include:
- Observation (files changed, line numbers)
- Logic chain (why each change was made)
- Caveats
- Conclusion
- Verification commands and full execution logs

Maintain your progress in `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_remediation_6_1\progress.md`.
When finished, send a coordination message to your parent (`ebf019cd-ccf0-44f1-97b0-e8b532cc9e09`).

