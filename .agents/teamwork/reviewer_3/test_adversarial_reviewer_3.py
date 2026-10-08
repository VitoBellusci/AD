import sys
import os
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.diffusion import (
    DiffusionScheduler,
    DiffusionForwardProcess,
    DiffusionReverseProcess,
)
from models.unet import Unet
from evaluate import sample_batch
from main import parse_args


def test_1_mathematical_inversion():
    """Verify exact algebraic roundtrip inversion: (x0, eps) -> v -> (rec_x0, rec_eps)."""
    print("[1/8] Testing mathematical inversion exactness across cosine and linear schedules...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    c, h, w = 3, 32, 32

    for sched_type in ["cosine", "linear"]:
        sched = DiffusionScheduler(num_time_steps=1000, schedule_type=sched_type, device=device)
        x0 = torch.randn(batch_size, c, h, w, device=device)
        noise = torch.randn(batch_size, c, h, w, device=device)
        
        # Test boundary timesteps as well as random timesteps
        test_ts = [0, 1, 250, 500, 750, 998, 999]
        for t_val in test_ts:
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            xt = (sched._extract(sched.sqrt_alpha_bars, t, x0) * x0 +
                  sched._extract(sched.sqrt_one_minus_alpha_bars, t, x0) * noise)
            
            v_target = sched.get_velocity(x0, noise, t)
            
            rec_x0 = sched.predict_x0_from_v(xt, v_target, t)
            rec_eps = sched.predict_noise_from_v(xt, v_target, t)
            
            x0_err = torch.max(torch.abs(rec_x0 - x0)).item()
            eps_err = torch.max(torch.abs(rec_eps - noise)).item()
            
            assert x0_err < 1e-5, f"Schedule '{sched_type}' at t={t_val}: x0 error too large: {x0_err}"
            assert eps_err < 1e-5, f"Schedule '{sched_type}' at t={t_val}: eps error too large: {eps_err}"
    print("   All mathematical roundtrips verified with error < 1e-5!")


def test_2_multiformat_timesteps_and_batch_mismatch():
    """Test int, float, list, tuple, 0D, 1D, 2D, 3D tensor timesteps and batch mismatch handling."""
    print("[2/8] Testing multi-format timesteps (list, tuple, 2D/3D tensor) and batch mismatch guards...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sched = DiffusionScheduler(num_time_steps=1000, device=device)
    rev = DiffusionReverseProcess(num_time_steps=1000, device=device)
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device).eval()

    x = torch.randn(2, 3, 16, 16, device=device)
    context = torch.randn(2, 4, 32, device=device)

    # Various formats that must succeed cleanly
    valid_formats = [
        ("int", 50),
        ("float", 50.0),
        ("list", [10, 20]),
        ("tuple", (10, 20)),
        ("0D tensor", torch.tensor(50, device=device)),
        ("1D length-1 tensor", torch.tensor([50], device=device)),
        ("1D batched tensor", torch.tensor([10, 20], device=device)),
        ("2D (1, 1) tensor", torch.tensor([[50]], device=device)),
        ("2D (2, 1) tensor", torch.tensor([[10], [20]], device=device)),
        ("3D (1, 1, 1) tensor", torch.tensor([[[50]]], device=device)),
    ]

    for name, t_in in valid_formats:
        c = sched._extract(sched.sqrt_alpha_bars, t_in, x)
        assert c.shape[0] in [1, 2] and c.shape[1:] == (1, 1, 1), f"Extract failed for {name}"
        res = c * x
        assert res.shape == x.shape, f"Broadcast failed for {name}"
        
        # Test sample with valid format
        out = rev.sample(unet, x, t_in, context=context, noise_free=True)
        assert out.shape == x.shape, f"Sample output shape mismatch for {name}: {out.shape}"
        print(f"   Format '{name}' passed cleanly.")

    # Batch mismatch guard test
    mismatched_t = torch.tensor([10, 20, 30], device=device)  # length 3 for batch size 2
    try:
        rev.sample(unet, x, mismatched_t)
        assert False, "Batch mismatch failed to raise ValueError!"
    except ValueError as e:
        print(f"   Batch mismatch caught with expected ValueError: {e}")


def test_3_dtype_preservation():
    """Verify dtype preservation across fp16, bf16, fp32, fp64."""
    print("[3/8] Testing tensor dtype preservation across fp16, fp32, fp64...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sched = DiffusionScheduler(num_time_steps=1000, device=device)
    rev = DiffusionReverseProcess(num_time_steps=1000, device=device)

    dtypes_to_test = [torch.float32, torch.float64]
    if device == "cuda":
        dtypes_to_test.extend([torch.float16, torch.bfloat16])

    for dt in dtypes_to_test:
        x = torch.randn(2, 3, 16, 16, device=device, dtype=dt)
        noise = torch.randn(2, 3, 16, 16, device=device, dtype=dt)
        v = torch.randn(2, 3, 16, 16, device=device, dtype=dt)
        t = torch.tensor([10, 20], device=device)

        v_calc = sched.get_velocity(x, noise, t)
        assert v_calc.dtype == dt, f"get_velocity dtype mismatch: got {v_calc.dtype}, expected {dt}"

        rec_x0 = sched.predict_x0_from_v(x, v, t)
        assert rec_x0.dtype == dt, f"predict_x0_from_v dtype mismatch: got {rec_x0.dtype}, expected {dt}"

        rec_eps = sched.predict_noise_from_v(x, v, t)
        assert rec_eps.dtype == dt, f"predict_noise_from_v dtype mismatch: got {rec_eps.dtype}, expected {dt}"

        # Mock model that returns velocity of matching dtype
        class MockModel(nn.Module):
            def forward(self, x, t, context=None, mask=None):
                return torch.zeros_like(x)

        model = MockModel()
        out = rev.sample(model, x, t, clip_denoised=True)
        assert out.dtype == dt, f"sample output dtype mismatch: got {out.dtype}, expected {dt}"
        print(f"   Dtype '{dt}' preserved across all operations.")


def test_4_terminal_step_langevin_isolation():
    """Verify that Langevin noise is 0 at t=0 even in mixed batches."""
    print("[4/8] Testing Langevin noise isolation at t=0 in uniform and mixed batches...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rev = DiffusionReverseProcess(num_time_steps=1000, device=device)

    class ZeroModel(nn.Module):
        def forward(self, x, t, context=None, mask=None):
            return torch.zeros_like(x)

    model = ZeroModel()
    torch.manual_seed(999)
    x = torch.randn(2, 3, 16, 16, device=device)

    # Uniform t=0
    t_zero = torch.tensor([0, 0], device=device)
    out1 = rev.sample(model, x, t_zero, noise_free=False)
    out2 = rev.sample(model, x, t_zero, noise_free=False)
    assert torch.allclose(out1, out2), "Uniform t=0 produced non-deterministic noise!"

    # Mixed batch: sample 0 at t=0, sample 1 at t=500
    t_mixed = torch.tensor([0, 500], device=device)
    torch.manual_seed(123)
    out_mixed1 = rev.sample(model, x, t_mixed, noise_free=False)
    torch.manual_seed(456)
    out_mixed2 = rev.sample(model, x, t_mixed, noise_free=False)

    # Item 0 (t=0) MUST be deterministic (0 Langevin noise)
    assert torch.allclose(out_mixed1[0], out_mixed2[0]), "Sample at t=0 in mixed batch received Langevin noise!"
    # Item 1 (t=500) MUST differ due to Langevin noise
    assert not torch.allclose(out_mixed1[1], out_mixed2[1]), "Sample at t=500 in mixed batch failed to receive noise!"
    print("   Langevin noise successfully isolated at t=0 in all cases.")


def test_5_reverse_process_sampling_variations():
    """Test reverse process sample with CFG, mask, and noise_free variations."""
    print("[5/8] Testing reverse process sample under various CFG and masking configurations...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rev = DiffusionReverseProcess(num_time_steps=1000, device=device)
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device).eval()

    x = torch.randn(2, 3, 16, 16, device=device)
    t = torch.tensor([50, 100], device=device)
    ctx = torch.randn(2, 4, 32, device=device)
    uncond_ctx = torch.randn(2, 4, 32, device=device)
    mask = torch.ones(2, 1, 1, 4, device=device)

    # CFG with mask
    out_cfg = rev.sample(unet, x, t, context=ctx, uncond_context=uncond_ctx, mask=mask, guidance_scale=3.5)
    assert out_cfg.shape == x.shape

    # CFG with mask=None
    out_cfg_nomask = rev.sample(unet, x, t, context=ctx, uncond_context=uncond_ctx, mask=None, guidance_scale=3.5)
    assert out_cfg_nomask.shape == x.shape

    # No CFG (guidance_scale=1.0)
    out_nocfg = rev.sample(unet, x, t, context=ctx, guidance_scale=1.0)
    assert out_nocfg.shape == x.shape

    # noise_free=True
    out_nf = rev.sample(unet, x, t, context=ctx, noise_free=True)
    assert out_nf.shape == x.shape

    # clip_denoised=False
    out_noclip = rev.sample(unet, x, t, context=ctx, clip_denoised=False)
    assert out_noclip.shape == x.shape
    print("   All reverse process CFG and masking variations verified.")


def test_6_ddim_sampling_loops():
    """Test DDIM sampling loop in evaluate.py with various step counts."""
    print("[6/8] Testing DDIM sampling loop in evaluate.py across step counts (1, 2, full)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rev = DiffusionReverseProcess(num_time_steps=1000, device=device)
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device).eval()

    ctx = torch.randn(2, 4, 32, device=device)
    uncond_ctx = torch.randn(2, 4, 32, device=device)
    mask = torch.ones(2, 1, 1, 4, device=device)

    for steps in [1, 2, 5]:
        out = sample_batch(
            unet=unet,
            reverse_process=rev,
            cond_ctx=ctx,
            uncond_ctx=uncond_ctx,
            mask=mask,
            shape=(2, 3, 16, 16),
            device=device,
            num_steps=steps,
            guidance_scale=3.0,
        )
        assert out.shape == (2, 3, 16, 16), f"DDIM sample_batch failed for num_steps={steps}"
        print(f"   DDIM {steps}-step sampling verified.")


def test_7_v_target_loss_and_backprop():
    """Test training loss computation with v_target and full backward pass."""
    print("[7/8] Testing v-target training loss computation and gradient backpropagation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fwd = DiffusionForwardProcess(num_time_steps=1000, device=device)
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device).train()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    images = torch.randn(2, 3, 16, 16, device=device)
    noise = torch.randn_like(images)
    timesteps = torch.randint(0, 1000, (2,), device=device).long()
    noisy_images = fwd.add_noise(images, noise, timesteps)
    context = torch.randn(2, 4, 32, device=device)

    optimizer.zero_grad()
    v_target = fwd.get_velocity(images, noise, timesteps)
    pred_v = unet(noisy_images, timesteps, context)
    loss = criterion(pred_v, v_target)
    loss.backward()

    # Verify gradients
    has_grads = any(p.grad is not None and torch.isfinite(p.grad).all() for p in unet.parameters())
    assert has_grads, "Gradients missing or non-finite!"
    optimizer.step()
    print(f"   Backpropagation verified: Loss={loss.item():.4f}, gradients valid.")


def test_8_cli_resilience():
    """Verify CLI argument parsing in train.py and main.py tolerates unknown args."""
    print("[8/8] Testing CLI argument parsing resilience with unknown args...")
    parser = parse_args()
    test_args = ["--max_steps", "2", "--local_rank", "0", "--extra_flag", "test"]
    parsed_args, unknown = parser.parse_known_args(test_args)
    assert parsed_args.max_steps == 2
    assert "--local_rank" in unknown
    print("   CLI argument parsing handles unknown args without SystemExit.")


if __name__ == "__main__":
    test_1_mathematical_inversion()
    test_2_multiformat_timesteps_and_batch_mismatch()
    test_3_dtype_preservation()
    test_4_terminal_step_langevin_isolation()
    test_5_reverse_process_sampling_variations()
    test_6_ddim_sampling_loops()
    test_7_v_target_loss_and_backprop()
    test_8_cli_resilience()
    print("\n=== ALL REVIEWER 3 ADVERSARIAL TESTS PASSED SUCCESSFULLY! ===")
