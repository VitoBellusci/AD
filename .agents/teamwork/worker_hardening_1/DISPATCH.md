## 2026-10-05T20:17:19Z
You are worker_hardening_1 (Hardening & Edge-Case Remediation Worker).
Your assigned working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_hardening_1

NOTE: DO NOT execute run_command. Use view_file and replace_file_content directly.

MANDATORY FIRST STEPS:
1. Initialize your BRIEFING.md, DISPATCH.md, and progress.md in your working directory.
2. Read ORIGINAL_REQUEST.md at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md.
3. Read the feedback reports from reviewer_code_1 and challenger_edge_cases_1:
   - c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\reviewer_code_1\handoff.md
   - c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\challenger_edge_cases_1\handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

SCOPE & EXCLUSIVE WRITE OWNERSHIP:
You own:
1. c:\Users\Admin\Desktop\avatar diffusion\models\transformer.py
2. c:\Users\Admin\Desktop\avatar diffusion\models\unet_parts.py
3. c:\Users\Admin\Desktop\avatar diffusion\inference.py
4. c:\Users\Admin\Desktop\avatar diffusion\evaluate.py
5. c:\Users\Admin\Desktop\avatar diffusion\train.py

TASKS TO EXECUTE:
1. In models/transformer.py:
   - In MultiHeadAttentionBlock.attention:
     Add rank-adaptive mask handling so 2D and 3D masks broadcast seamlessly to [B, Heads, Seq_Len, Seq_Len]:
     ```python
     if mask is not None:
         if mask.ndim == 2:
             mask = mask.unsqueeze(1).unsqueeze(2)
         elif mask.ndim == 3:
             mask = mask.unsqueeze(1)
         attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
     ```
   - In FullTextEncoder.forward, Encoder.forward, and EncoderBlock.forward:
     Ensure mask has default None: `def forward(self, x, mask=None):` for backward compatibility.

2. In models/unet_parts.py:
   - In SpatialCrossAttention.forward:
     Add nan_to_num defense-in-depth after computing out:
     ```python
     if torch.isnan(out).any():
         out = torch.nan_to_num(out, nan=0.0)
     ```

3. In inference.py:
   - Line 131: Replace `uncond_mask = (uncond_tokens != pad_token_id)...` with:
     ```python
     uncond_mask = torch.ones_like(mask)
     ```
     (Because uncond_tokens is 100% pad tokens, checking != pad_token_id produced all False, passing an all-False mask during CFG. Unconditional context must attend to tokens with all True!).

4. In evaluate.py:
   - In evaluate.py line 163 & 168:
     Ensure `mask = (text_tokens != pad_id).unsqueeze(1).unsqueeze(2).to(device)`
     and `uncond_mask = torch.ones_like(mask)`.

5. In train.py:
   - When CFG null condition dropout is triggered, ensure mask is set to `torch.ones_like(mask)` so unconditional pass does not use text token mask.

6. VERIFICATION:
   Inspect all 5 files with view_file to confirm clean syntax, correct signatures, and exact logic.
   Write your handoff report to: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_hardening_1\handoff.md.
   Notify parent via send_message when done.
