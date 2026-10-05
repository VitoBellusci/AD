## 2026-10-05T18:59:49Z
You are worker_models_1 (Architecture & Mask Propagation Worker).
Your assigned working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_models_1

MANDATORY FIRST STEPS:
1. Initialize your BRIEFING.md, DISPATCH.md, and progress.md in your working directory.
2. Read ORIGINAL_REQUEST.md at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md.
3. Read Section 10 (specifically 10.1, Blueprint 1.3 and Blueprint 1.4) of audit_report.md at: c:\Users\Admin\Desktop\avatar diffusion\audit_report.md.
4. Also read worker_foundation_1 handoff at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_foundation_1\handoff.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

SCOPE & EXCLUSIVE WRITE OWNERSHIP:
You own exclusively:
1. c:\Users\Admin\Desktop\avatar diffusion\models\transformer.py
2. c:\Users\Admin\Desktop\avatar diffusion\models\unet_parts.py
3. c:\Users\Admin\Desktop\avatar diffusion\models\unet.py

TASKS TO EXECUTE:
1. In models/transformer.py (Blueprint 1.3, DEF-03, DEF-15):
   - In MultiHeadAttentionBlock.attention:
     Replace -1e-9 mask fill with float("-inf"):
     attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
     Add safe nan_to_num handling if an entire row is masked:
     if torch.isnan(attention_scores).any():
         attention_scores = torch.nan_to_num(attention_scores, nan=0.0)
   - Ensure custom LayerNormalization (or ResidualConnection) uses proper parameter shapes (DEF-15).

2. In models/unet_parts.py (Blueprint 1.4, DEF-04):
   - In SpatialCrossAttention.forward(self, x, context, mask=None):
     Implement dynamic rank-adaptive mask handling to prevent shape mismatch:
     if mask is not None:
         if mask.ndim == 2:
             attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, Seq_Len]
         elif mask.ndim == 3:
             attn_mask = mask.unsqueeze(1) # [B, 1, 1, Seq_Len]
         elif mask.ndim == 4:
             attn_mask = mask
         else:
             attn_mask = mask.view(b, 1, 1, -1)
         if attn_mask.dtype != torch.bool:
             attn_mask = (attn_mask != 0)
     Pass attn_mask to F.scaled_dot_product_attention.

3. In models/unet.py (Blueprint 1.4, DEF-04):
   - In Unet.forward(self, x, time, context, mask=None):
     Add mask=None parameter to forward signature.
     Explicitly propagate mask=mask to every cross-attention module:
     - skip1 = self.attn_inc(self.inc(x, t), context, mask=mask)
     - skip2 = self.attn_down1(self.down1(skip1, t), context, mask=mask)
     - skip3 = self.attn_down2(x_down2, context, mask=mask)
     - bott = self.attn_bott1(bott, context, mask=mask)
     - x = self.attn_up1(x, context, mask=mask)
     - x = self.attn_up2(x, context, mask=mask)

4. VERIFICATION & TESTS:
   - Run python verification script/command to test:
     a. MultiHeadAttentionBlock attention with mask and check that masked positions are exactly 0.0 and no NaNs occur.
     b. SpatialCrossAttention with 2D [2, 20], 3D [2, 1, 20], and 4D [2, 1, 1, 20] masks, verifying no shape errors.
     c. Unet forward pass with x [2, 3, 64, 64], t [2], context [2, 20, 128], mask [2, 20], verifying forward pass succeeds and output is [2, 3, 64, 64].
     d. Unet forward pass with mask=None to ensure backward compatibility.
   - Record test commands and exact outputs.

5. DELIVERABLE:
   Write your handoff report to: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_models_1\handoff.md.
   The report must include Observation, Logic Chain, Caveats, Conclusion, and Verification Method with executed commands and output.
   Notify parent when done.
