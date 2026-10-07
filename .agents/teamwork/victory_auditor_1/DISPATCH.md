## 2026-10-07T13:22:24Z
[Dispatch from parent ad807ab8-25b4-4d59-8ac2-15a40352ec65]
Modify the avatar diffusion codebase to use natural language descriptive prompts instead of numerical attribute IDs, matching the project requirements.
Working directory: c:\Users\Admin\Desktop\avatar diffusion
Integrity mode: development

Requirements:
R1. Natural Language Caption Generation: Modify preprocessing/caption_generator.py to map dataset numerical metadata values to meaningful natural language descriptors without numerical IDs.
R2. Update Inference Prompts: Update inference.py default prompt and ood_prompts to use natural language rather than numerical IDs.
R3. No Terminal Execution: Do not execute any terminal commands. Static code analysis only.
Acceptance Criteria:
- preprocessing/caption_generator.py is updated and maps metadata to strings without numerical IDs.
- inference.py uses only natural language prompts for OOD and defaults.
- The code is syntactically valid and type-safe.
