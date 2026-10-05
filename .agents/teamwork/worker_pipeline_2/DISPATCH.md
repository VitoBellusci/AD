# Dispatch for Pipeline Integration Worker
## 2026-10-05T17:26:08Z
You are the Pipeline Integration & Usability Worker.
Your working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_pipeline_2
You MUST read ORIGINAL_REQUEST.md located at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md before starting work.
Authoritative defect audit report is at: c:\Users\Admin\Desktop\avatar diffusion\audit_report.md (specifically Section 10, Blueprints 2.2, 2.3, 2.4, 3.1, 3.2, 3.3 and Sections 7, 8, 9 regarding DEF-06, DEF-07, DEF-08, DEF-10, DEF-11, DEF-12, DEF-14, DEF-18, DEF-19, DEF-20).
Also read prior handoffs:
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_foundation_1\handoff.md
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_3\handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Write Ownership:
You own exclusively:
1. c:\Users\Admin\Desktop\avatar diffusion\main.py
2. c:\Users\Admin\Desktop\avatar diffusion\train.py
3. c:\Users\Admin\Desktop\avatar diffusion\inference.py
4. c:\Users\Admin\Desktop\avatar diffusion\evaluate.py

Tasks:
1. main.py:
   - Fix line 54 unpack bug: unpack 4 partitions from splitter.split(): train_idx, val_idx, test_ind_idx, ood_idx = splitter.split(raw_metadata).
   - Instantiate validation dataset and DataLoader from val_idx, pass val_loader into train() (DEF-10).
   - Support CLI arguments or config for conditional / unconditional baseline (DEF-11).
   - Implement deterministic checkpoint resolution via regex integer epoch sorting (Blueprint 3.2, DEF-20), removing fragile os.path.getctime.
   - Ensure configure_optimizers adheres to Blueprint 3.1 (weight decay for 2D/4D weights, 0.0 for 1D biases/norms; LinearLR warmup + CosineAnnealingLR).

2. train.py:
   - In train(): support val_loader parameter, evaluate validation loss at each epoch, and report it (DEF-10).
   - Fix mask unassigned error: if not conditional, ensure mask = None so predicted_noise = unet(noisy_images, timesteps, context, mask=mask) runs cleanly without UnboundLocalError.
   - Ensure decoupled gradient clipping (DEF-19) is retained: clip unet and text_encoder separately.
   - Add AMP support with torch.cuda.amp.autocast and GradScaler if device is CUDA (DEF-14).

3. inference.py:
   - Eliminate hardcoded crash on missing checkpoint_epoch_22.pt (DEF-08). Use resolve_checkpoint(checkpoint_dir="checkpoints", explicit_path=args.checkpoint) with regex epoch sorting. If no checkpoint exists and none provided, handle gracefully (e.g. exit with clear message or initialize random model for testing).
   - Eliminate blocking while True: input(...) loop (DEF-12). Replace with argparse CLI arguments per Blueprint 3.3 (--prompt, --seed, --guidance_scale, --checkpoint, --num_steps, --batch_size, --output_dir, --test_ood).
   - Implement accelerated sampling / DDIM step spacing support when --num_steps < 1000 so generation does not force all 1000 steps (DEF-12).
   - Make OOD testing reachable via --test_ood flag.
   - Ensure load_vocab() is handled safely if vocab.json does not exist yet.

4. evaluate.py:
   - Fix AvatarDataset instantiation at line 52: AvatarDataset in preprocessing/dataset.py expects (image_paths, metadata_list, config, tokenizer). In evaluate.py, load raw images and metadata (matching dataset.py load_data) and construct AvatarDataset properly.
   - Ensure reverse_process.sample() arguments match models/diffusion.py (which now takes model, x, t, context, uncond_context, mask, uncond_mask, guidance_scale, noise_free, clip_denoised).
   - Ensure evaluate.py runs standalone evaluation on both test_ind and test_ood splits without throwing exceptions.

5. Test & Verify:
   - Test importing all four modules.
   - Test python inference.py --help and python evaluate.py --help.
   - Test running preprocessing, single batch training/eval step, and verify no exceptions are raised.
   - Document commands executed and test results in handoff.md.
6. Write c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_pipeline_2\handoff.md and notify parent when done.
