import torch
import pytest
from ecg_cvae.losses import morphology_loss, loss_fn

def test_morphology_loss_basic():
    """Ensure morphology_loss returns 0 when original == reconstructed"""
    x = torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.0]]])  # batch=1, seq_len=5
    recon_x = x.clone()
    loss = morphology_loss(x, recon_x, prominence_threshold=0.1)
    assert loss == 0.0

def test_morphology_loss_detects_difference():
    """Check morphology_loss detects peak differences"""
    x = torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.0]]])
    recon_x = torch.tensor([[[0.0, 0.5, 0.0, 0.5, 0.0]]])
    loss = morphology_loss(x, recon_x, prominence_threshold=0.1)
    assert loss > 0.0

def test_loss_fn_returns_tensors():
    """Ensure loss_fn returns 4 torch tensors"""
    x = torch.randn(2, 1, 10)
    recon_x = x + 0.1 * torch.randn_like(x)
    mu = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)
    total_loss, recon_loss, kld_loss, morph_loss = loss_fn(recon_x, x, mu, logvar, weight_morph=1.0)
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(recon_loss, torch.Tensor)
    assert isinstance(kld_loss, torch.Tensor)
    assert isinstance(morph_loss, torch.Tensor)
    assert total_loss.shape == ()
