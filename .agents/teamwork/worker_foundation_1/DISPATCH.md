## 2026-10-05T17:19:36Z
You are the Foundation Remediation Worker.
Your working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_foundation_1
You MUST read ORIGINAL_REQUEST.md located at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md before starting work.
Authoritative defect audit report is at: c:\Users\Admin\Desktop\avatar diffusion\audit_report.md (specifically Section 10, Blueprint 1.1 and Blueprint 2.1).
Also read Explorer handoffs:
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_1\handoff.md
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_2\handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Write Ownership:
You own exclusively:
1. c:\Users\Admin\Desktop\avatar diffusion\preprocessing\preprocessing_config.json
2. c:\Users\Admin\Desktop\avatar diffusion\preprocessing\config.py
3. c:\Users\Admin\Desktop\avatar diffusion\models\diffusion.py

Tasks:
1. In preprocessing/preprocessing_config.json:
   Update "splits_path" to "preprocessing/splits.json" and "ood_blocked_combinations" to [[["hair", "98"], ["glasses", "11"]]] as specified in Blueprint 1.1.
2. In preprocessing/config.py:
   Add self.splits_path = raw_config.get("splits_path", "preprocessing/splits.json") to PreprocessingConfig.__init__.
3. In models/diffusion.py:
   Apply Blueprint 2.1 in full:
   - In DiffusionScheduler.__init__: ensure alphas_cumprod_prev, posterior_mean_coef1, posterior_mean_coef2 are properly computed and registered on self.device.
   - In DiffusionReverseProcess.sample: implement the complete Blueprint 2.1 method:
     Signature: sample(self, model, x, t, context=None, uncond_context=None, mask=None, uncond_mask=None, guidance_scale=3.0, noise_free=False, clip_denoised=True)
     Support CFG with mask concatenation (handling mask_input when mask/uncond_mask provided).
     Properly define beta_t = self.betas[t].to(x.device)[:, None, None, None] (fixing the fatal NameError).
     Compute pred_x0 and clip it to [-1.0, 1.0] if clip_denoised is True.
     Compute mean using posterior_mean_coef1 and posterior_mean_coef2.
     Inject noise with sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20)) * z.
4. Verify your changes by running verification commands (e.g. testing PreprocessingConfig loading, checking that splitter generates > 0 OOD samples on dataset metadata, testing sample() forward step with dummy tensors on CPU).
5. Document all code edits, commands executed, and verification outputs in c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_foundation_1\handoff.md.
6. Send a completion message to parent when done.
