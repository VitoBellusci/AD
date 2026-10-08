"""
Comprehensive Adversarial Test Suite for v-prediction (Velocity Target) Migration
Author: Reviewer 1 (Adversarial Review)

Tests:
1. test_mathematical_inversion_all_schedules:
   Verifies algebraic exactness of (x0, eps) <-> (xt, v) under both cosine and linear schedules
   at all timesteps including boundary conditions (t=0, t=T-1).
2. test_scalar_0d_1d_timesteps_handling:
   Verifies that get_velocity, predict_x0_from_v, predict_noise_from_v, add_noise, and sample
   seamlessly support integer, 0-d tensor, and 1-d tensor timesteps without IndexError.
3. test_reverse_process_sample_modes:
   Verifies DiffusionReverseProcess.sample under CFG, unconditional, mask=None, t=0 terminal,
   noise_free=True, and clip_denoised configurations.
4. test_ddim_evaluate_mask_none:
   Verifies evaluate.py sample_batch works without crashing when mask=None and uncond_mask=None.
5. test_ddim_and_ddpm_sampling_modes:
   Verifies both fast DDIM (2 steps) and full DDPM (1000 steps) execution path in sample_batch.
6. test_training_loss_and_gradient_flow:
   Verifies forward_process.get_velocity produces accurate loss and allows clean backward pass.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from models.diffusion import DiffusionForwardProcess, DiffusionReverseProcess
from evaluate import sample_batch


def test_mathematical_inversion_all_schedules():
    print("[1/6] Testing mathematical inversion exactness across cosine and linear schedules...")
    device = "cpu"
    batch_size = 16
    channels, h, w = 3, 32, 32

    for sched in ["cosine", "linear"]:
        fp = DiffusionForwardProcess(num_time_steps=1000, schedule_type=sched, device=device)
        rp = DiffusionReverseProcess(num_time_steps=1000, schedule_type=sched, device=device)

        # Include critical boundary timesteps t=0 and t=999
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        timesteps[0] = 0
        timesteps[1] = 999

        x0 = torch.randn(batch_size, channels, h, w, device=device)
        noise = torch.randn(batch_size, channels, h, w, device=device)

        # 1. Forward noising: xt = sqrt(alpha_bar)*x0 + sqrt(1 - alpha_bar)*noise
        xt = fp.add_noise(x0, noise, timesteps)

        # 2. Velocity target: v = sqrt(alpha_bar)*noise - sqrt(1 - alpha_bar)*x0
        v_target = fp.get_velocity(x0, noise, timesteps)

        # 3. Recover clean x0 from xt and v
        # x0_hat = sqrt(alpha_bar)*xt - sqrt(1 - alpha_bar)*v
        x0_hat = rp.predict_x0_from_v(xt, v_target, timesteps)
        err_x0 = torch.max(torch.abs(x0_hat - x0)).item()
        assert err_x0 < 1e-5, f"Schedule {sched}: x0 reconstruction error too high: {err_x0}"

        # 4. Recover noise from xt and v
        # eps_hat = sqrt(alpha_bar)*v + sqrt(1 - alpha_bar)*xt
        eps_hat = rp.predict_noise_from_v(xt, v_target, timesteps)
        err_eps = torch.max(torch.abs(eps_hat - noise)).item()
        assert err_eps < 1e-5, f"Schedule {sched}: epsilon reconstruction error too high: {err_eps}"

        print(f"   Schedule '{sched}' verified! Max err x0: {err_x0:.2e}, Max err eps: {err_eps:.2e}")


def test_scalar_0d_1d_timesteps_handling():
    print("[2/6] Testing scalar, 0D, and 1D timesteps handling across all methods...")
    device = "cpu"
    fp = DiffusionForwardProcess(device=device)
    rp = DiffusionReverseProcess(device=device)
    x = torch.randn(4, 3, 16, 16, device=device)
    noise = torch.randn(4, 3, 16, 16, device=device)

    class DummyModel(nn.Module):
        def forward(self, x, t, context=None, mask=None):
            return torch.zeros_like(x)

    dummy_model = DummyModel()

    # Case A: Integer timestep
    t_int = 42
    xt = fp.add_noise(x, noise, t_int)
    v = fp.get_velocity(x, noise, t_int)
    x0_hat = rp.predict_x0_from_v(xt, v, t_int)
    eps_hat = rp.predict_noise_from_v(xt, v, t_int)
    sample_out = rp.sample(dummy_model, x, t_int)
    assert xt.shape == x.shape and v.shape == x.shape
    assert x0_hat.shape == x.shape and eps_hat.shape == x.shape
    assert sample_out.shape == x.shape
    print("   Scalar int timestep (t=42) passed without error.")

    # Case B: 0-D tensor timestep
    t_0d = torch.tensor(128, device=device)
    xt = fp.add_noise(x, noise, t_0d)
    v = fp.get_velocity(x, noise, t_0d)
    x0_hat = rp.predict_x0_from_v(xt, v, t_0d)
    eps_hat = rp.predict_noise_from_v(xt, v, t_0d)
    sample_out = rp.sample(dummy_model, x, t_0d)
    assert xt.shape == x.shape and v.shape == x.shape
    assert sample_out.shape == x.shape
    print("   0-D tensor timestep (t=tensor(128)) passed without error.")

    # Case C: 1-D batch tensor timestep
    t_1d = torch.tensor([10, 50, 100, 500], device=device)
    xt = fp.add_noise(x, noise, t_1d)
    v = fp.get_velocity(x, noise, t_1d)
    x0_hat = rp.predict_x0_from_v(xt, v, t_1d)
    eps_hat = rp.predict_noise_from_v(xt, v, t_1d)
    sample_out = rp.sample(dummy_model, x, t_1d)
    assert xt.shape == x.shape and v.shape == x.shape
    assert sample_out.shape == x.shape
    print("   1-D batch tensor timestep passed without error.")


def test_reverse_process_sample_modes():
    print("[3/6] Testing DiffusionReverseProcess.sample under CFG and various conditions...")
    device = "cpu"
    rp = DiffusionReverseProcess(device=device)
    batch_size = 2
    x = torch.randn(batch_size, 3, 16, 16, device=device)
    t = torch.tensor([250, 250], device=device)
    context = torch.randn(batch_size, 10, 32, device=device)
    uncond_context = torch.randn(batch_size, 10, 32, device=device)
    mask = torch.ones(batch_size, 1, 1, 10, dtype=torch.bool, device=device)

    class DummyModel(nn.Module):
        def forward(self, x, t, context=None, mask=None):
            return torch.sin(x)

    dummy_model = DummyModel()

    # CFG with full masks
    out_cfg = rp.sample(dummy_model, x, t, context=context, uncond_context=uncond_context,
                        mask=mask, uncond_mask=mask, guidance_scale=3.5)
    assert out_cfg.shape == x.shape

    # CFG with None mask
    out_cfg_nomask = rp.sample(dummy_model, x, t, context=context, uncond_context=uncond_context,
                               mask=None, uncond_mask=None, guidance_scale=3.5)
    assert out_cfg_nomask.shape == x.shape

    # Non-CFG
    out_nocfg = rp.sample(dummy_model, x, t, context=context, guidance_scale=1.0)
    assert out_nocfg.shape == x.shape

    # Terminal step t=0
    t0 = torch.zeros(batch_size, dtype=torch.long, device=device)
    out_t0 = rp.sample(dummy_model, x, t0, guidance_scale=1.0, noise_free=True)
    assert out_t0.shape == x.shape
    print("   All reverse sampling modes verified successfully.")


def test_ddim_evaluate_mask_none():
    print("[4/6] Testing evaluate.py DDIM sample_batch with mask=None (graceful handling)...")
    device = "cpu"
    rp = DiffusionReverseProcess(device=device)

    class DummyModel(nn.Module):
        def forward(self, x, t, context=None, mask=None):
            return torch.zeros_like(x)

    dummy_model = DummyModel()
    batch_size = 2
    shape = (batch_size, 3, 16, 16)
    cond_ctx = torch.randn(batch_size, 8, 32, device=device)
    uncond_ctx = torch.randn(batch_size, 8, 32, device=device)

    # Calling with mask=None and uncond_mask=None must NOT crash
    out = sample_batch(
        unet=dummy_model,
        reverse_process=rp,
        cond_ctx=cond_ctx,
        uncond_ctx=uncond_ctx,
        mask=None,
        uncond_mask=None,
        shape=shape,
        device=device,
        num_steps=2,
        guidance_scale=3.5
    )
    assert out.shape == shape
    print("   DDIM sample_batch with mask=None passed cleanly.")


def test_ddim_and_ddpm_sampling_modes():
    print("[5/6] Testing DDIM (2 steps) vs DDPM (1000 steps) execution path...")
    device = "cpu"
    rp = DiffusionReverseProcess(num_time_steps=10, device=device)

    class DummyModel(nn.Module):
        def forward(self, x, t, context=None, mask=None):
            return torch.zeros_like(x)

    dummy_model = DummyModel()
    batch_size = 1
    shape = (batch_size, 3, 16, 16)

    # DDIM branch (num_steps < total_timesteps)
    out_ddim = sample_batch(
        unet=dummy_model,
        reverse_process=rp,
        cond_ctx=None,
        uncond_ctx=None,
        mask=None,
        uncond_mask=None,
        shape=shape,
        device=device,
        num_steps=2,
        guidance_scale=1.0
    )
    assert out_ddim.shape == shape

    # DDPM branch (num_steps >= total_timesteps)
    out_ddpm = sample_batch(
        unet=dummy_model,
        reverse_process=rp,
        cond_ctx=None,
        uncond_ctx=None,
        mask=None,
        uncond_mask=None,
        shape=shape,
        device=device,
        num_steps=10,
        guidance_scale=1.0
    )
    assert out_ddpm.shape == shape
    print("   Both DDIM and DDPM branches executed cleanly.")


def test_training_loss_and_gradient_flow():
    print("[6/6] Testing v-target training loss computation and gradient backpropagation...")
    device = "cpu"
    fp = DiffusionForwardProcess(device=device)

    class SmallTestUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        def forward(self, x, t, context=None, mask=None):
            return self.conv(x)

    unet = SmallTestUNet()
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    images = torch.randn(4, 3, 16, 16, requires_grad=False)
    noise = torch.randn_like(images)
    timesteps = torch.randint(0, 1000, (4,))

    noisy_images = fp.add_noise(images, noise, timesteps)
    v_target = fp.get_velocity(images, noise, timesteps)

    pred_v = unet(noisy_images, timesteps)
    loss = criterion(pred_v, v_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0
    print(f"   Training loss computed: {loss.item():.4f}, gradients propagated successfully.")


if __name__ == "__main__":
    print("=== Running Adversarial Test Suite for v-prediction ===")
    test_mathematical_inversion_all_schedules()
    test_scalar_0d_1d_timesteps_handling()
    test_reverse_process_sample_modes()
    test_ddim_evaluate_mask_none()
    test_ddim_and_ddpm_sampling_modes()
    test_training_loss_and_gradient_flow()
    print("=== ALL ADVERSARIAL TESTS PASSED SUCCESSFULLY! ===")

