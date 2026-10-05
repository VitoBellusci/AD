## 2026-10-05T19:58:12Z
From: parent (752b9482-f249-49b5-8219-37fe369ea6ea)
To: worker_eval_infer_1

You are worker_eval_infer_1 (Evaluation & Inference Usability Worker).
Your assigned working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_eval_infer_1

NOTE: DO NOT execute run_command. Use view_file and replace_file_content directly. Do not launch interactive commands.

MANDATORY FIRST STEPS:
1. Initialize your BRIEFING.md, DISPATCH.md, and progress.md in your working directory.
2. Read ORIGINAL_REQUEST.md at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md.
3. Read Section 10 (specifically Blueprints 2.2, 2.3, 2.4, 3.2, 3.3, and Sections 7/8/9 regarding DEF-06, DEF-07, DEF-08, DEF-12, DEF-18, DEF-20) of audit_report.md at: c:\Users\Admin\Desktop\avatar diffusion\audit_report.md.
4. Also read worker_foundation_1 handoff, worker_preprocessing_1 handoff, and worker_models_2 handoff in .agents/teamwork/.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

SCOPE & EXCLUSIVE WRITE OWNERSHIP:
You own exclusively:
1. c:\Users\Admin\Desktop\avatar diffusion\metrics.py
2. c:\Users\Admin\Desktop\avatar diffusion\evaluate.py
3. c:\Users\Admin\Desktop\avatar diffusion\inference.py

TASKS TO EXECUTE:
1. In metrics.py (Blueprint 2.2, Blueprint 2.4, DEF-07, DEF-18):
   - Add static method `_ensure_zero_one_range(images: torch.Tensor) -> torch.Tensor`:
     ```python
     @staticmethod
     def _ensure_zero_one_range(images: torch.Tensor) -> torch.Tensor:
         if images.min() < 0.0:
             images = (images + 1.0) / 2.0
         return torch.clamp(images, 0.0, 1.0)
     ```
   - In `update_quality_metrics(self, real_images: torch.Tensor, fake_images: torch.Tensor)`:
     Independently normalize real and fake images:
     ```python
     real_norm = self._ensure_zero_one_range(real_images)
     fake_norm = self._ensure_zero_one_range(fake_images)
     self.fid.update(real_norm, real=True)
     self.fid.update(fake_norm, real=False)
     self.kid.update(real_norm, real=True)
     self.kid.update(fake_norm, real=False)
     ```
   - Add `AttributeAlignmentEvaluator` class (Blueprint 2.4):
     ```python
     class AttributeAlignmentEvaluator:
         def __init__(self, classifier_model, device):
             self.classifier = classifier_model.to(device).eval()
         def evaluate_alignment(self, images, expected_attributes):
             with torch.no_grad():
                 preds = self.classifier(images)
                 correct = (preds == expected_attributes).float().mean()
             return correct.item()
     ```

2. In evaluate.py (Blueprint 2.3, DEF-06, DEF-20):
   - Implement standalone batch-accumulating evaluation pipeline:
     - In `resolve_checkpoint`: use regex integer epoch extraction `re.search(r'checkpoint_epoch_(\d+)\.pt', os.path.basename(p))`.
     - Load dataset properly using AvatarDataset(data_dir=data_dir, config=config, tokenizer=tokenizer) or matching dataset.py load_data.
     - Load `splits.json` from config.splits_path (evaluating both "Ordinary Test (In-Distribution)" from test_ind and "OOD Test (Compositional Held-Out)" from test_ood).
     - Accumulate batches into `evaluator.update_quality_metrics(real_images, fake_images)` across batches BEFORE calling `evaluator.compute_quality_metrics()`.
     - Reverse process sampling call: pass `context=cond_ctx, uncond_context=uncond_ctx, mask=mask, uncond_mask=uncond_mask, guidance_scale=3.5, clip_denoised=True`.
     - Support CLI arguments: `--checkpoint`, `--batch_size`, `--num_samples`, `--data_dir`.

3. In inference.py (Blueprint 3.2, Blueprint 3.3, DEF-08, DEF-12, DEF-20):
   - Replace hardcoded `checkpoint_path = "checkpoints/checkpoint_epoch_22.pt"` crash (DEF-08).
     Implement `resolve_checkpoint(checkpoint_dir="checkpoints", explicit_path=args.checkpoint)` with regex epoch sorting. If checkpoints directory is empty and no explicit path is given, handle gracefully without unhandled FileNotFoundError (e.g. exit with clear error or allow test initialization).
   - Replace blocking `while True: input(...)` loop with CLI `argparse` arguments (DEF-12, Blueprint 3.3):
     `--prompt`, `--seed`, `--guidance_scale`, `--checkpoint`, `--num_steps`, `--batch_size`, `--output_dir`, `--test_ood`.
   - Implement accelerated / DDIM sampling step spacing when `--num_steps < 1000` (e.g. 50 steps) so inference is rapid and not forced to 1000 steps.
   - Make OOD combination testing reachable via `--test_ood` flag.
   - Guard `tokenizer.load_vocab()` gracefully if `vocab.json` is missing.

4. VERIFICATION:
   Inspect metrics.py, evaluate.py, and inference.py using view_file to confirm changes are accurate, robust, and introduce zero syntax errors or regressions.
   Construct a clear static verification and test spec in your handoff report.

5. DELIVERABLE:
   Write your handoff report to: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_eval_infer_1\handoff.md.
   The report must include Observation, Logic Chain, Caveats, Conclusion, and Verification Method.
   Notify parent via send_message when done.
