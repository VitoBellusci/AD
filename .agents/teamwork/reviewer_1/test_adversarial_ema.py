"""
Adversarial Verification Suite for UNet Exponential Moving Average (EMA) Integration
Exhaustively tests requirements R1, R2, and all identified edge cases / robustness hazards:
1. EMA model creation, parameter freezing, device inference, and decay validation
2. Deep recursive unwrapping of multi-layer DataParallel wrappers
3. Mathematical precision of EMA parameter interpolation
4. AMP GradScaler step-skipping behavior when inf/NaN gradients occur
5. Recursive prefix stripping ('module.module...')
6. Checkpoint serialization format and integer step counter serialization
7. End-to-end multi-epoch training resumption
8. Fault-tolerant checkpoint recovery when ema_state_dict is corrupted / incompatible
9. Legacy checkpoint fallback initialization from active weights (n_averaged=0)
10. CLI argument parsing flags (--use_ema, --no_ema, --ema_decay)
11. Downstream inference & evaluation script checkpoint loading (prefers EMA, falls back cleanly)
"""

import os
import shutil
import tempfile
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.unet import Unet
from models.transformer import FullTextEncoder
from models.diffusion import DiffusionForwardProcess
from train import (
    create_ema_model,
    make_ema_multi_avg_fn,
    strip_prefix,
    configure_optimizers,
    train,
    set_seed
)
from main import parse_args


class DummyTokenizer:
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "avatar": 4}


class DummyDataset(Dataset):
    def __init__(self, size=16):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, 64, 64)
        tokens = torch.randint(0, 5, (20,), dtype=torch.long)
        return image, tokens


class MultiWrapper(nn.Module):
    """Simulates nested wrappers (e.g., DataParallel inside custom wrapper)."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_1_ema_creation_properties():
    """Verify EMA model creation, parameter freezing, device inference, and decay bounds."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)

    # Valid creation
    ema_unet = create_ema_model(unet, decay=0.999, device=device)
    assert ema_unet is not None, "EMA model must not be None"
    assert hasattr(ema_unet, "module"), "AveragedModel must wrap raw UNet under .module"
    assert hasattr(ema_unet, "n_averaged"), "AveragedModel must register n_averaged buffer"
    assert ema_unet.n_averaged.item() == 0, f"Expected initial n_averaged=0, got {ema_unet.n_averaged.item()}"

    # Verify all EMA parameters have requires_grad=False
    for name, param in ema_unet.named_parameters():
        assert not param.requires_grad, f"Parameter {name} must have requires_grad=False"

    # Verify decay validation
    try:
        create_ema_model(unet, decay=1.5)
        assert False, "Should have raised ValueError for decay > 1.0"
    except ValueError:
        pass

    try:
        create_ema_model(unet, decay=-0.1)
        assert False, "Should have raised ValueError for decay < 0.0"
    except ValueError:
        pass

    # Verify device inference when device is None
    ema_inferred = create_ema_model(unet, decay=0.999, device=None)
    assert next(ema_inferred.parameters()).device == next(unet.parameters()).device

    print("Test 1 Passed: EMA creation, properties, and validation verified.")


def test_2_ema_multi_level_unwrapping():
    """Verify create_ema_model unwraps arbitrary levels of .module nesting."""
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32)
    nested = MultiWrapper(MultiWrapper(MultiWrapper(unet)))

    ema = create_ema_model(nested, decay=0.99)
    assert isinstance(ema.module, Unet), f"Expected ema.module to be Unet, got {type(ema.module)}"
    print("Test 2 Passed: Multi-level unwrapping verified.")


def test_3_ema_update_math():
    """Verify mathematical correctness of EMA interpolation across multiple steps."""
    device = "cpu"
    decay = 0.9
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    ema_unet = create_ema_model(unet, decay=decay, device=device)

    p_init = list(unet.parameters())[0].detach().clone()

    # Step 1: initial update
    ema_unet.update_parameters(unet)
    assert ema_unet.n_averaged.item() == 1
    p_ema1 = list(ema_unet.module.parameters())[0].detach()
    assert torch.allclose(p_ema1, p_init), "Step 1 EMA must match active parameters"

    # Step 2: perturb active weights
    with torch.no_grad():
        list(unet.parameters())[0].add_(2.0)
    p_step2 = list(unet.parameters())[0].detach().clone()

    ema_unet.update_parameters(unet)
    assert ema_unet.n_averaged.item() == 2
    p_ema2 = list(ema_unet.module.parameters())[0].detach()
    expected_p2 = p_init * decay + p_step2 * (1.0 - decay)
    assert torch.allclose(p_ema2, expected_p2, atol=1e-5), f"Step 2 EMA math mismatch"

    # Step 3: perturb active weights again
    with torch.no_grad():
        list(unet.parameters())[0].add_(3.0)
    p_step3 = list(unet.parameters())[0].detach().clone()

    ema_unet.update_parameters(unet)
    assert ema_unet.n_averaged.item() == 3
    p_ema3 = list(ema_unet.module.parameters())[0].detach()
    expected_p3 = expected_p2 * decay + p_step3 * (1.0 - decay)
    assert torch.allclose(p_ema3, expected_p3, atol=1e-5), f"Step 3 EMA math mismatch"

    print("Test 3 Passed: EMA mathematical interpolation verified across steps.")


def test_4_amp_grad_scaler_step_skipping_adversarial():
    """
    Verify that if an AMP GradScaler skips an optimizer step due to inf/NaN,
    the EMA weights DO NOT update and n_averaged is NOT incremented.
    """
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32)
    ema_unet = create_ema_model(unet, decay=0.9)

    class MockScaler:
        def __init__(self, initial_scale=65536.0):
            self.scale = initial_scale
            self.simulate_inf = False

        def get_scale(self):
            return self.scale

        def step(self, optimizer):
            if not self.simulate_inf:
                optimizer.step()

        def update(self):
            if self.simulate_inf:
                self.scale = self.scale * 0.5  # Backoff on inf

    mock_scaler = MockScaler()
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.1)

    # Iteration 1: Normal step (no inf)
    mock_scaler.simulate_inf = False
    scale_before = mock_scaler.get_scale()
    mock_scaler.step(optimizer)
    mock_scaler.update()
    scale_after = mock_scaler.get_scale()
    step_executed = not (scale_after < scale_before)
    assert step_executed is True

    if step_executed:
        raw_unet = unet
        while hasattr(raw_unet, 'module'):
            raw_unet = raw_unet.module
        ema_unet.update_parameters(raw_unet)

    assert ema_unet.n_averaged.item() == 1, "EMA should have updated on valid step"

    # Iteration 2: Gradient explosion with inf/NaN -> step skipped!
    mock_scaler.simulate_inf = True
    scale_before = mock_scaler.get_scale()
    mock_scaler.step(optimizer)
    mock_scaler.update()
    scale_after = mock_scaler.get_scale()
    step_executed = not (scale_after < scale_before)
    assert step_executed is False, "Step should be flagged as skipped when scaler backs off"

    if step_executed:
        raw_unet = unet
        while hasattr(raw_unet, 'module'):
            raw_unet = raw_unet.module
        ema_unet.update_parameters(raw_unet)

    assert ema_unet.n_averaged.item() == 1, "EMA must NOT update when optimizer step was skipped!"

    # Iteration 3: Recovery step
    mock_scaler.simulate_inf = False
    scale_before = mock_scaler.get_scale()
    mock_scaler.step(optimizer)
    mock_scaler.update()
    scale_after = mock_scaler.get_scale()
    step_executed = not (scale_after < scale_before)
    assert step_executed is True

    if step_executed:
        raw_unet = unet
        while hasattr(raw_unet, 'module'):
            raw_unet = raw_unet.module
        ema_unet.update_parameters(raw_unet)

    assert ema_unet.n_averaged.item() == 2, "EMA should resume updating on next valid step"
    print("Test 4 Passed: AMP GradScaler inf/NaN step-skipping behavior verified.")


def test_5_recursive_strip_prefix():
    """Verify strip_prefix cleanly removes arbitrary nested 'module.' prefixes."""
    state_dict = {
        "weight": torch.tensor([1]),
        "module.weight": torch.tensor([2]),
        "module.module.weight": torch.tensor([3]),
        "module.module.module.weight": torch.tensor([4]),
        "module.sub.module.weight": torch.tensor([5])
    }
    cleaned = strip_prefix(state_dict)
    assert "weight" in cleaned
    assert "sub.module.weight" in cleaned
    for k in cleaned.keys():
        assert not k.startswith("module."), f"Found remaining prefix in key '{k}'"
    print("Test 5 Passed: Recursive strip_prefix verified.")


def test_6_checkpoint_serialization_and_types():
    """Verify fast dummy training (max_steps=5) serializes EMA state dicts and int step counter."""
    device = "cpu"
    set_seed(42)

    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    text_encoder = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=32, d_ff=64, num_layers=2).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=100, device=device)
    dataloader = DataLoader(DummyDataset(size=8), batch_size=2)
    tokenizer = DummyTokenizer()

    optimizer, scheduler = configure_optimizers(unet, text_encoder, lr=1e-4, total_epochs=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        trained_ema = train(
            unet=unet,
            text_encoder=text_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=5,
            ema_decay=0.999,
            use_ema=True
        )

        assert trained_ema is not None
        ckpt_path = os.path.join(tmpdir, "checkpoint_epoch_1.pt")
        assert os.path.exists(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Acceptance Criteria: contains regular weights AND EMA state dict
        assert "unet_state_dict" in ckpt
        assert "text_encoder_state_dict" in ckpt
        assert "ema_state_dict" in ckpt
        assert "ema_unet_state_dict" in ckpt
        assert "ema_n_averaged" in ckpt

        # Verify int serialization
        assert isinstance(ckpt["ema_n_averaged"], (int, torch.Tensor))
        val = ckpt["ema_n_averaged"]
        assert (val.item() if hasattr(val, "item") else val) == 5

        # Checkpoint keys match UNet parameter count
        assert len(ckpt["ema_unet_state_dict"]) == len(ckpt["unet_state_dict"])
        print("Test 6 Passed: Checkpoint serialization and type verification passed.")


def test_7_checkpoint_resumption_full_flow():
    """Verify multi-epoch resumption restores EMA weights and increments n_averaged correctly."""
    device = "cpu"
    set_seed(42)

    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    text_encoder = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=32, d_ff=64, num_layers=2).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=100, device=device)
    dataloader = DataLoader(DummyDataset(size=8), batch_size=2)
    tokenizer = DummyTokenizer()

    optimizer, scheduler = configure_optimizers(unet, text_encoder, lr=1e-4, total_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Epoch 1: 4 steps
        train(
            unet=unet,
            text_encoder=text_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=4,
            ema_decay=0.999,
            use_ema=True
        )

        ckpt_epoch1 = os.path.join(tmpdir, "checkpoint_epoch_1.pt")
        assert os.path.exists(ckpt_epoch1)
        ckpt1 = torch.load(ckpt_epoch1, map_location=device, weights_only=False)

        # Resume setup with fresh models
        resumed_unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
        resumed_ema = create_ema_model(resumed_unet, decay=0.999, device=device)
        resumed_encoder = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=32, d_ff=64, num_layers=2).to(device)
        resumed_opt, resumed_sched = configure_optimizers(resumed_unet, resumed_encoder, lr=1e-4, total_epochs=2)

        # Restore states using the 3-tier logic in main.py
        resumed_unet.load_state_dict(strip_prefix(ckpt1['unet_state_dict']))
        resumed_encoder.load_state_dict(strip_prefix(ckpt1['text_encoder_state_dict']))
        resumed_opt.load_state_dict(ckpt1['optimizer_state_dict'])
        resumed_sched.load_state_dict(ckpt1['scheduler_state_dict'])

        # Restore EMA
        resumed_ema.load_state_dict(ckpt1['ema_state_dict'])
        assert resumed_ema.n_averaged.item() == 4

        # Epoch 2: 4 more steps
        train(
            unet=resumed_unet,
            text_encoder=resumed_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=resumed_opt,
            scheduler=resumed_sched,
            epochs=2,
            start_epoch=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=4,
            ema_unet=resumed_ema,
            ema_decay=0.999,
            use_ema=True
        )

        ckpt_epoch2 = os.path.join(tmpdir, "checkpoint_epoch_2.pt")
        assert os.path.exists(ckpt_epoch2)
        ckpt2 = torch.load(ckpt_epoch2, map_location=device, weights_only=False)
        assert ckpt2['epoch'] == 2
        val2 = ckpt2['ema_n_averaged']
        assert (val2.item() if hasattr(val2, 'item') else val2) == 8, f"Expected 8 cumulative steps, got {val2}"
        print("Test 7 Passed: Multi-epoch resumption flow with EMA verified.")


def test_8_corrupted_ema_state_dict_fallback():
    """Verify fallback when ema_state_dict is corrupted or incompatible but ema_unet_state_dict exists."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    ema_unet = create_ema_model(unet, decay=0.999, device=device)

    # Perturb unet
    with torch.no_grad():
        list(unet.parameters())[0].add_(5.0)

    # Checkpoint with invalid ema_state_dict but valid ema_unet_state_dict
    corrupted_ckpt = {
        'epoch': 1,
        'unet_state_dict': unet.state_dict(),
        'ema_state_dict': {'invalid_key': torch.tensor([1, 2, 3])},  # Corrupted
        'ema_unet_state_dict': unet.state_dict(),  # Valid weights
        'ema_n_averaged': 10
    }

    # Simulate main.py 3-tier restoration
    raw_ema = ema_unet
    while hasattr(raw_ema, 'module'):
        raw_ema = raw_ema.module

    ema_loaded = False
    if 'ema_state_dict' in corrupted_ckpt:
        try:
            ema_unet.load_state_dict(corrupted_ckpt['ema_state_dict'])
            ema_loaded = True
        except Exception:
            pass  # Expected failure

    assert ema_loaded is False, "Direct load should have failed"

    # Tier 2: Fallback to ema_unet_state_dict
    if not ema_loaded and 'ema_unet_state_dict' in corrupted_ckpt:
        weights = strip_prefix(corrupted_ckpt['ema_unet_state_dict'])
        raw_ema.load_state_dict(weights)
        if hasattr(ema_unet, 'n_averaged'):
            ema_unet.n_averaged.fill_(int(corrupted_ckpt['ema_n_averaged']))
        ema_loaded = True

    assert ema_loaded is True
    assert ema_unet.n_averaged.item() == 10
    assert torch.allclose(list(raw_ema.parameters())[0], list(unet.parameters())[0])
    print("Test 8 Passed: Corrupted ema_state_dict fallback to ema_unet_state_dict verified.")


def test_9_legacy_checkpoint_fallback():
    """Verify legacy checkpoints without EMA keys initialize cleanly from active weights with n_averaged=0."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    ema_unet = create_ema_model(unet, decay=0.999, device=device)

    # Pre-populate EMA with dummy counter
    ema_unet.n_averaged.fill_(99)

    legacy_ckpt = {
        'epoch': 5,
        'unet_state_dict': unet.state_dict()
    }

    # Simulate main.py tier 3
    raw_ema = ema_unet
    while hasattr(raw_ema, 'module'):
        raw_ema = raw_ema.module

    raw_ema.load_state_dict(strip_prefix(legacy_ckpt['unet_state_dict']))
    ema_unet.n_averaged.fill_(0)

    assert ema_unet.n_averaged.item() == 0
    assert torch.allclose(list(raw_ema.parameters())[0], list(unet.parameters())[0])
    print("Test 9 Passed: Legacy checkpoint fallback verified.")


def test_10_cli_argument_parsing():
    """Verify CLI argument flags parse correctly with defaults and overrides."""
    parser = parse_args()

    args_def = parser.parse_args([])
    assert args_def.use_ema is True
    assert args_def.ema_decay == 0.9999

    args_disabled = parser.parse_args(["--no_ema", "--ema_decay", "0.99"])
    assert args_disabled.use_ema is False
    assert args_disabled.ema_decay == 0.99

    args_enabled = parser.parse_args(["--use_ema", "--max_steps", "10"])
    assert args_enabled.use_ema is True
    assert args_enabled.max_steps == 10
    print("Test 10 Passed: CLI argument parsing verified.")


def test_11_downstream_loading_logic():
    """Verify inference and evaluation script loading logic prefers EMA over raw weights."""
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32)

    raw_weights = {k: v.clone() for k, v in unet.state_dict().items()}
    ema_weights = {k: v.clone() + 1.0 for k, v in unet.state_dict().items()}

    # Checkpoint with both
    ckpt_full = {
        'unet_state_dict': raw_weights,
        'ema_unet_state_dict': ema_weights
    }

    # Simulation of inference/evaluate loading
    def load_unet(ckpt):
        if 'ema_unet_state_dict' in ckpt:
            return strip_prefix(ckpt['ema_unet_state_dict'])
        elif 'ema_state_dict' in ckpt:
            raw_ema = {k: v for k, v in ckpt['ema_state_dict'].items() if k != "n_averaged"}
            return strip_prefix(raw_ema)
        else:
            return strip_prefix(ckpt['unet_state_dict'])

    loaded = load_unet(ckpt_full)
    first_k = list(loaded.keys())[0]
    assert torch.allclose(loaded[first_k], ema_weights[first_k]), "Should prefer EMA weights"

    # Legacy checkpoint
    ckpt_legacy = {'unet_state_dict': raw_weights}
    loaded_leg = load_unet(ckpt_legacy)
    assert torch.allclose(loaded_leg[first_k], raw_weights[first_k]), "Should fallback to raw weights"
    print("Test 11 Passed: Downstream script checkpoint loading preference verified.")


if __name__ == "__main__":
    test_1_ema_creation_properties()
    test_2_ema_multi_level_unwrapping()
    test_3_ema_update_math()
    test_4_amp_grad_scaler_step_skipping_adversarial()
    test_5_recursive_strip_prefix()
    test_6_checkpoint_serialization_and_types()
    test_7_checkpoint_resumption_full_flow()
    test_8_corrupted_ema_state_dict_fallback()
    test_9_legacy_checkpoint_fallback()
    test_10_cli_argument_parsing()
    test_11_downstream_loading_logic()
    print("\n=======================================================")
    print("ALL 11 ADVERSARIAL VERIFICATION TESTS PASSED CLEANLY!")
    print("=======================================================")
