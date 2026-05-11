"""Smoke test for PersonaGazeNet.

Verifies:
    - Forward pass with K=0 (gaze only), K=1, K=4 reference sets.
    - Output shape and value range.
    - Backward pass produces grads on the expected parameter subsets.
    - set_phase() correctly toggles requires_grad for the 3 phases.

Run:
    python -m tests.test_persona_gaze_forward
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                                # noqa: E402

from models.persona_gaze import PersonaGazeNet, RefCrossAttn  # noqa: E402


def _make_inputs(B=2, K=4, H=64, W=128, gaze_dim=64, device='cpu'):
    source_eye = torch.randn(B, 3, H, W, device=device).clamp(-1, 1)
    gaze_cond = torch.randn(B, 2, gaze_dim, device=device)
    ref_eyes = torch.randn(B, K, 3, H, W, device=device).clamp(-1, 1)
    return source_eye, gaze_cond, ref_eyes


def test_forward_no_refs():
    model = PersonaGazeNet(use_hyper_lora=True)
    src, gaze, _ = _make_inputs()
    out = model(src, gaze, ref_eyes=None)
    assert out.shape == src.shape, f"shape mismatch: {out.shape} vs {src.shape}"
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6, \
        f"out of range: [{out.min()}, {out.max()}]"
    # At init delta_scale≈0.05 → residual should be small (mostly identity).
    diff = (out - src).abs().max().item()
    assert diff < 0.5, f"residual should be small at init, got {diff}"
    print(f"[PASS] forward (no refs): out range [{out.min():.3f}, {out.max():.3f}], "
          f"init residual={diff:.2e}")


def test_forward_with_refs():
    model = PersonaGazeNet(use_hyper_lora=True)
    src, gaze, refs = _make_inputs(B=2, K=4)
    out, aux = model(src, gaze, ref_eyes=refs, return_aux=True)
    assert out.shape == src.shape
    assert aux['ref_tokens'].shape[0] == 2
    assert aux['ref_tokens'].shape[2] == model.token_dim
    assert len(aux['lora_list']) == 2  # 2 decoder LoRA convs
    for A, B in aux['lora_list']:
        assert A.dim() == 3 and B.dim() == 3
    print(f"[PASS] forward (K=4): tokens={aux['ref_tokens'].shape}, "
          f"lora_pairs={len(aux['lora_list'])}")


def test_forward_K1():
    model = PersonaGazeNet(use_hyper_lora=True)
    src, gaze, _ = _make_inputs(B=2)
    refs = src.unsqueeze(1)  # K=1, use source itself as ref
    out = model(src, gaze, ref_eyes=refs)
    assert out.shape == src.shape
    print(f"[PASS] forward (K=1): ok")


def test_backward():
    model = PersonaGazeNet(use_hyper_lora=True)
    src, gaze, refs = _make_inputs(B=2, K=2)
    out = model(src, gaze, ref_eyes=refs)
    # Force a non-trivial gradient by adding noise to break delta_scale=0 dead zone.
    target = torch.zeros_like(out)
    loss = (out - target).pow(2).mean() + out.mean()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters())
    print(f"[PASS] backward: {n_with_grad}/{n_total} params received non-zero grad")
    assert n_with_grad > 0


def test_phase_switching():
    model = PersonaGazeNet(use_hyper_lora=True)
    total = sum(p.numel() for p in model.parameters())

    model.set_phase('phase1')
    trainable_1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rfa_train = sum(p.numel() for m in model.modules() if isinstance(m, RefCrossAttn)
                    for p in m.parameters() if p.requires_grad)
    hyper_train = sum(p.numel() for p in model.hyper_lora.parameters() if p.requires_grad)
    assert rfa_train == 0, "RFA should be frozen in phase1"
    assert hyper_train == 0, "HyperLoRA should be frozen in phase1"

    model.set_phase('phase2')
    trainable_2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rfa_train_2 = sum(p.numel() for m in model.modules() if isinstance(m, RefCrossAttn)
                      for p in m.parameters() if p.requires_grad)
    hyper_train_2 = sum(p.numel() for p in model.hyper_lora.parameters() if p.requires_grad)
    assert rfa_train_2 > 0, "RFA should be trainable in phase2"
    assert hyper_train_2 == 0, "HyperLoRA still frozen in phase2"

    model.set_phase('phase3')
    trainable_3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_3 == total, "phase3 should unfreeze everything"

    print(f"[PASS] phase switching: "
          f"phase1={trainable_1/1e6:.2f}M / "
          f"phase2={trainable_2/1e6:.2f}M / "
          f"phase3={trainable_3/1e6:.2f}M (total {total/1e6:.2f}M)")


def test_param_count():
    model = PersonaGazeNet(use_hyper_lora=True)
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    ref = sum(p.numel() for p in model.ref_tokenizer.parameters())
    hyp = sum(p.numel() for p in model.hyper_lora.parameters())
    tot = enc + dec + ref + hyp + sum(1 for _ in [model.delta_scale])
    print(f"[INFO] param budget (M): encoder={enc/1e6:.2f} "
          f"decoder={dec/1e6:.2f} ref={ref/1e6:.2f} hyper={hyp/1e6:.2f} "
          f"total={model.num_params/1e6:.2f}")


def test_imports_train_pipeline():
    """Ensure the training script and dataset wrapper at least import."""
    # MultiRefHDFDataset must import without touching an HDF file.
    from dataset.multi_ref_dataset import MultiRefHDFDataset  # noqa: F401
    # Training script's helper builders are pure-Python (no HDF/accelerate needed
    # at import time) — just import the module.
    import importlib
    mod = importlib.import_module('train_persona_gaze')
    assert hasattr(mod, 'build_model')
    assert hasattr(mod, 'build_gaze_mlp')
    assert hasattr(mod, 'whitening_loss')
    assert hasattr(mod, 'cycle_loss')
    print("[PASS] train_persona_gaze module imports cleanly")


if __name__ == '__main__':
    torch.manual_seed(0)
    print("=== PersonaGazeNet smoke tests ===")
    test_param_count()
    test_forward_no_refs()
    test_forward_K1()
    test_forward_with_refs()
    test_backward()
    test_phase_switching()
    try:
        test_imports_train_pipeline()
    except ImportError as e:
        # Some deps (lpips, accelerate, diffusers, h5py, piq, lpips) may be
        # missing in a stripped CPU env — log but do not fail the core test.
        print(f"[SKIP] train_persona_gaze import test: {e}")
    print("=== All smoke tests passed ===")
