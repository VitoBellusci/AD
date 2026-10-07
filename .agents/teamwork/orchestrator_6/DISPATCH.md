# Orchestrator Dispatch

## Working Directory
`c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_6`

## Request Source
`c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md` (specifically the request under header `## 2026-10-07T13:55:14Z`)

## Task Objective
Perform a comprehensive final code review, functional verification, and fix-up of the Avatar Diffusion project. Ensure it strictly meets all assignment requirements, is completely bug-free, and is fully ready for the definitive training run.

## Requirements
- **R1. Preprocessing and Compositional Split**: Verify and fix the data preprocessing pipeline. It must resize avatars, normalize consistently, and generate deterministic multi-attribute captions. It must construct a vocabulary *only* from the training split. It must correctly implement a "compositional split" (holding out specific attribute combinations from training).
- **R2. From-Scratch Models**: Ensure that the text tokenizer, text encoder (2-4 layer Transformer), and U-Net denoiser (pixel-space DDPM) are built and initialized entirely from scratch. No pre-trained models (like CLIP, T5, Stable Diffusion, pretrained VAEs) or black-box Diffusers training pipelines are allowed.
- **R3. Diffusion Components and Conditioning**: Ensure the implementation includes a correct text-conditioning mechanism (e.g., cross-attention), noise schedule, forward noising process, reverse sampling loop, and checkpointing logic.
- **R4. Evaluation Metrics**: Ensure the evaluation code correctly calculates and logs image quality metrics (FID or KID), diversity across seeds, parameter count, sampling time, and memory usage, on both ordinary and compositional out-of-distribution prompts.

## Acceptance Criteria
- [ ] No pre-trained checkpoints or text encoders are imported or downloaded in the codebase.
- [ ] The text encoder and U-Net are instantiated as custom PyTorch modules or raw configurations without pre-trained weights.
- [ ] The compositional split logic correctly isolates at least one specific attribute combination from the training set.
- [ ] A short dummy training run (e.g., 1-2 epochs on a tiny subset) executes from start to finish without crashing.
- [ ] The reverse sampling loop successfully generates a batch of images from text prompts without runtime errors.
- [ ] Evaluation metrics (FID/KID, parameter count) are successfully computed without crashing.


## 2026-10-07T13:57:04Z

You are orchestrator_6, the Project Orchestrator for the Avatar Diffusion project.
Your working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_6

Your dispatch brief and task requirements are in:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_6\DISPATCH.md
The full verbatim user request history is in:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md (specifically the newest request under ## 2026-10-07T13:55:14Z)

Mission:
Perform a comprehensive final code review, functional verification, and fix-up of the Avatar Diffusion project. Ensure it strictly meets all assignment requirements, is completely bug-free, and is fully ready for the definitive training run.

Requirements:
R1. Preprocessing and Compositional Split: Verify and fix the data preprocessing pipeline. It must resize avatars, normalize consistently, and generate deterministic multi-attribute captions. It must construct a vocabulary *only* from the training split. It must correctly implement a "compositional split" (holding out specific attribute combinations from training).
R2. From-Scratch Models: Ensure that the text tokenizer, text encoder (2-4 layer Transformer), and U-Net denoiser (pixel-space DDPM) are built and initialized entirely from scratch. No pre-trained models (like CLIP, T5, Stable Diffusion, pretrained VAEs) or black-box Diffusers training pipelines are allowed.
R3. Diffusion Components and Conditioning: Ensure the implementation includes a correct text-conditioning mechanism (e.g., cross-attention), noise schedule, forward noising process, reverse sampling loop, and checkpointing logic.
R4. Evaluation Metrics: Ensure the evaluation code correctly calculates and logs image quality metrics (FID or KID), diversity across seeds, parameter count, sampling time, and memory usage, on both ordinary and compositional out-of-distribution prompts.

Acceptance Criteria:
- No pre-trained checkpoints or text encoders are imported or downloaded in the codebase.
- The text encoder and U-Net are instantiated as custom PyTorch modules or raw configurations without pre-trained weights.
- The compositional split logic correctly isolates at least one specific attribute combination from the training set.
- A short dummy training run (e.g., 1-2 epochs on a tiny subset) executes from start to finish without crashing.
- The reverse sampling loop successfully generates a batch of images from text prompts without runtime errors.
- Evaluation metrics (FID/KID, parameter count) are successfully computed without crashing.

Initialize your BRIEFING.md, plan.md, and progress.md in your working directory immediately, assemble your team roster, and execute. Update your progress.md regularly so Sentinel crons can monitor your progress. When completely finished and verified, send a message to Sentinel reporting project completion.
