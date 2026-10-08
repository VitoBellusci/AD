"""
Unit and Integration Test Suite for v-prediction (Velocity Target) Migration
Tests:
- Mathematical exactness of v-target calculation and inversion to x0 and epsilon
- DiffusionReverseProcess.sample with model velocity prediction
- DDIM sampling in evaluate.py and inference.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from models.diffusion import DiffusionForwardProcess, DiffusionReverseProcess
from models.unet import Unet
from models.transformer import FullTextEncoder

def test_mathematical_v_parameterization():
    """
    Verifies that:
    1. v_target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * original
    2. pred_x0 = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * v
    3. pred_eps = sqrt_alpha_bar_t * v + sqrt_one_minus_alpha_bar_t * x_t
    exactly invert to original and noise for all timesteps.
    """
    device = "cpu"
    forward_process = DiffusionForwardProcess(num_time_steps=1000, schedule_type="cosine", device=device)
    reverse_process = DiffusionReverseProcess(num_time_steps=1000, schedule_type="cosine", device=device)

    batch_size = 8
    channels, h, w = 3, 64, 64
    x0 = torch.randn(batch_size, channels, h, w, device=device)
    noise = torch.randn(batch_size, channels, h, w, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)

    # 1. Forward noising: x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    xt = forward_process.add_noise(x0, noise, timesteps)

    # 2. Velocity target: v = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x_0
    sqrt_alpha_bar_t = forward_process.sqrt_alpha_bars[timesteps][:, None, None, None]
    sqrt_one_minus_alpha_bar_t = forward_process.sqrt_one_minus_alpha_bars[timesteps][:, None, None, None]
    v_target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x0

    # Also test get_velocity helper
    v_helper = forward_process.get_velocity(x0, noise, timesteps)
    assert torch.allclose(v_target, v_helper, atol=1e-6), "get_velocity mismatch"

    # 3. Recover x0 from (xt, v_target)
    # x0 = sqrt_alpha_bar_t * xt - sqrt_one_minus_alpha_bar_t * v
    pred_x0 = sqrt_alpha_bar_t * xt - sqrt_one_minus_alpha_bar_t * v_target
    err_x0 = torch.max(torch.abs(pred_x0 - x0)).item()
    assert torch.allclose(pred_x0, x0, atol=1e-5), f"Failed to reconstruct x0: max error = {err_x0}"

    # Also test predict_x0_from_v helper
    pred_x0_helper = reverse_process.predict_x0_from_v(xt, v_target, timesteps)
    assert torch.allclose(pred_x0, pred_x0_helper, atol=1e-6), "predict_x0_from_v mismatch"

    # 4. Recover epsilon from (xt, v_target)
    # eps = sqrt_alpha_bar_t * v + sqrt_one_minus_alpha_bar_t * xt
    pred_eps = sqrt_alpha_bar_t * v_target + sqrt_one_minus_alpha_bar_t * xt
    err_eps = torch.max(torch.abs(pred_eps - noise)).item()
    assert torch.allclose(pred_eps, noise, atol=1e-5), f"Failed to reconstruct epsilon: max error = {err_eps}"

    # Also test predict_noise_from_v helper
    pred_eps_helper = reverse_process.predict_noise_from_v(xt, v_target, timesteps)
    assert torch.allclose(pred_eps, pred_eps_helper, atol=1e-6), "predict_noise_from_v mismatch"

    print("Mathematical exactness verified! Max x0 error:", err_x0, "Max eps error:", err_eps)

def test_reverse_process_sample():
    """
    Verifies that DiffusionReverseProcess.sample executes properly with a dummy model
    outputting velocity.
    """
    device = "cpu"
    reverse_process = DiffusionReverseProcess(num_time_steps=1000, schedule_type="cosine", device=device)
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    t = torch.tensor([500, 500], dtype=torch.long, device=device)
    context = torch.randn(batch_size, 20, 64, device=device)
    uncond_context = torch.randn(batch_size, 20, 64, device=device)

    # Dummy model predicting velocity with matching shape
    class DummyVelocityModel(nn.Module):
        def forward(self, x, t, context=None, mask=None):
            return torch.zeros_like(x)

    dummy_model = DummyVelocityModel()

    # CFG sample
    out_cfg = reverse_process.sample(
        dummy_model, x, t,
        context=context, uncond_context=uncond_context,
        guidance_scale=3.5, noise_free=False, clip_denoised=True
    )
    assert out_cfg.shape == x.shape, f"CFG sample output shape mismatch: {out_cfg.shape}"

    # Non-CFG sample
    out_nocfg = reverse_process.sample(
        dummy_model, x, t,
        context=context,
        guidance_scale=1.0, noise_free=False, clip_denoised=True
    )
    assert out_nocfg.shape == x.shape, f"Non-CFG sample output shape mismatch: {out_nocfg.shape}"

    # Terminal step t=0
    t0 = torch.zeros(batch_size, dtype=torch.long, device=device)
    out_t0 = reverse_process.sample(
        dummy_model, x, t0,
        guidance_scale=1.0, noise_free=True, clip_denoised=True
    )
    assert out_t0.shape == x.shape, f"t=0 sample output shape mismatch: {out_t0.shape}"

    print("ReverseProcess.sample verified!")

def test_ddim_sampling_evaluate():
    """
    Verifies that sample_batch in evaluate.py runs without error with v-prediction.
    """
    from evaluate import sample_batch
    device = "cpu"
    reverse_process = DiffusionReverseProcess(num_time_steps=1000, schedule_type="cosine", device=device)

    class DummyVelocityModel(nn.Module):
        def forward(self, x, t, context=None, mask=None):
            return torch.randn_like(x)

    model = DummyVelocityModel()
    batch_size = 2
    shape = (batch_size, 3, 32, 32)
    cond_ctx = torch.randn(batch_size, 10, 64, device=device)
    uncond_ctx = torch.randn(batch_size, 10, 64, device=device)
    mask = torch.ones(batch_size, 1, 1, 10, dtype=torch.bool, device=device)
    uncond_mask = torch.ones_like(mask)

    out = sample_batch(
        unet=model,
        reverse_process=reverse_process,
        cond_ctx=cond_ctx,
        uncond_ctx=uncond_ctx,
        mask=mask,
        uncond_mask=uncond_mask,
        shape=shape,
        device=device,
        num_steps=2,
        guidance_scale=3.5
    )
    assert out.shape == shape, f"sample_batch output shape mismatch: {out.shape}"
    print("DDIM sample_batch in evaluate.py verified!")

if __name__ == "__main__":
    test_mathematical_v_parameterization()
    test_reverse_process_sample()
    test_ddim_sampling_evaluate()
    print("All unit tests passed successfully!")
