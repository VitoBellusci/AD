## 2026-10-07T13:59:24Z
You are spec_miner_survey_6_1.
Your working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\spec_miner_survey_6_1
Your parent is orchestrator_6 (conversation ID: ebf019cd-ccf0-44f1-97b0-e8b532cc9e09).

MANDATORY FIRST STEP: Read the user request history in:
c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md
Pay special attention to the latest prompt under ## 2026-10-07T13:55:14Z and the background requests.

OBJECTIVE:
Perform a comprehensive specification mining audit of all requirements, architectural constraints, hyperparameters, from-scratch rules, evaluation metrics, and compositional split rules.

SOURCES OF TRUTH TO INVESTIGATE:
1. c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md
2. c:\Users\Admin\Desktop\avatar diffusion\pdf_content.txt (and Deep_Learning_2026_VI 1.pdf context)
3. c:\Users\Admin\Desktop\avatar diffusion\audit_report.md
4. c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_6\DISPATCH.md

EXTRACT AND SYSTEMATIZE:
1. All assignment requirements (R1: Preprocessing & Compositional Split, R2: From-Scratch Models, R3: Diffusion & Conditioning, R4: Evaluation Metrics).
2. Explicit constraints: What is forbidden (pretrained models, CLIP, T5, VAEs, Diffusers pipelines, etc.) and what is required (Tiny parameter budget, custom modules, pixel-space DDPM, etc.).
3. Compositional split specification: What attributes, combinations, holdout rules, train/val/test/OOD partition rules exist.
4. Model architecture specifications: U-Net channels/blocks, text encoder depth (2-4 layers), cross-attention conditioning, DDPM timesteps/betas/noise schedule.
5. Evaluation metrics specifications: FID/KID, diversity across seeds, parameter count, sampling time, GPU/CPU memory usage, evaluation on both IID and OOD prompts.
6. Acceptance criteria and testable conditions.

SCOPE BOUNDARIES:
- Read-only specification mining. Do NOT modify any source code files.
- Deliver your findings in: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\spec_miner_survey_6_1\handoff.md
- Maintain your liveness in: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\spec_miner_survey_6_1\progress.md

When finished, send a coordination message to your parent (ebf019cd-ccf0-44f1-97b0-e8b532cc9e09).
