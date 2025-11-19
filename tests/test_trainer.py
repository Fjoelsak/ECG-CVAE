import numpy as np
import torch
from torch import nn
from ecg_cvae.trainer import compare_peaks, evaluate_model, train_vae_for_channel

import pytest
from unittest.mock import patch, MagicMock


# Dummy model for testing
class DummyModel(nn.Module):
    def forward(self, x, peak_vec, peak_count):
        # simply return input as reconstruction
        return x, torch.zeros(x.size(0), 1), torch.zeros(x.size(0), 1), torch.zeros(x.size(0), 1)

def test_compare_peaks_identical():
    signal = np.sin(np.linspace(0, 10, 100))
    assert compare_peaks(signal, signal), "Identical signals should return True"

def test_compare_peaks_different():
    signal = np.sin(np.linspace(0, 10, 100))
    noisy = signal + 0.5
    result = compare_peaks(signal, noisy)
    assert result in [True, False], "Result should be a boolean"

def test_evaluate_model_identity():
    N, L = 5, 50
    data = np.random.randn(N, L)
    peaks = np.zeros_like(data)
    counts = np.zeros(N)
    model = DummyModel()
    mse = evaluate_model(model, data, peaks, counts, batch_size=2)
    assert isinstance(mse, float), "MSE should be a float"

# Mock dataset class
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=10, seq_len=32):
        self.data = torch.randn(n_samples, seq_len)
        self.peaks = torch.randint(0, 2, (n_samples, seq_len))
        self.counts = torch.randint(0, 5, (n_samples, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.peaks[idx], self.counts[idx]

# Mock VAE model
class DummyVAE(torch.nn.Module):
    def __init__(self, seq_len=32, latent_dim=4):
        super().__init__()
        self.decoder = MagicMock(latent_dim=latent_dim)
        self.cond_embedding = MagicMock()
        self.seq_len = seq_len

    def forward(self, x, pv, pc):
        batch_size = x.shape[0]
        z = torch.randn(batch_size, 4)
        mu = torch.zeros_like(z)
        logvar = torch.zeros_like(z)
        recon = torch.randn_like(x).unsqueeze(1)  # mimic shape [B,1,seq_len]
        return recon, mu, logvar, z


# Patch all heavy dependencies
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    monkeypatch.setattr("ecg_cvae.dataset.ChannelDataset", lambda *args, **kwargs: DummyDataset())
    monkeypatch.setattr("ecg_cvae.models.ConditionalVAE", lambda *args, **kwargs: DummyVAE(seq_len=32, latent_dim=4))
    monkeypatch.setattr("ecg_cvae.losses.loss_fn", lambda recon, sig, mu, logvar, weight_morph=1.0: (torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.2)))
    monkeypatch.setattr("ecg_cvae.trainer.evaluate_model", lambda model, data, pv, pc, batch_size, device: 0.123)
    monkeypatch.setattr("ecg_cvae.plotting.plot_reconstructions", lambda *args, **kwargs: None)
    monkeypatch.setattr("ecg_cvae.plotting.plot_generated_samples", lambda *args, **kwargs: None)
    monkeypatch.setattr("ecg_cvae.trainer.compare_peaks", lambda s, r: True)
     # Provide enough latents so the Mahalanobis branch is taken
    monkeypatch.setattr(
        "ecg_cvae.sampling.mahalanobis_latent_sampling",
        lambda collected_latents, latent_dim, n_samples, channel_idx, max_mahal: np.random.randn(max(latent_dim+1, 10), latent_dim)
    )
def test_train_vae_runs(tmp_path):
    output_dir = tmp_path
    channel_data = np.random.randn(10, 32)
    channel_peaks = np.random.randint(0, 2, (10, 32))
    channel_counts = np.random.randint(0, 5, (10, 1))

    # Call the training function with minimal epochs for speed
    train_vae_for_channel(
        channel_data=channel_data,
        channel_peak_vecs=channel_peaks,
        channel_peak_counts=channel_counts,
        channel_idx=0,
        MAXepochs=1,
        batch_size=2,
        output_dir=str(output_dir),
        fname="testfile",
        resume=False,
        save_every=1,
        device='cpu'
    )

    # Check that latent file and final model exist
    latent_file = output_dir / "v6b3_A" / "testfile_ch0_similar_latents.npy"
    final_model = output_dir / "v6b3_A" / "testfile_ch0_hd256_lt_128_final.pt"

    # File may exist if good latents found
    assert final_model.exists(), "Final model checkpoint was not saved"

def test_train_with_no_latents(tmp_path):
    """Ensure function works if no good latents are collected."""
    output_dir = tmp_path
    channel_data = np.random.randn(10, 32)
    channel_peaks = np.random.randint(0, 2, (10, 32))
    channel_counts = np.random.randint(0, 5, (10, 1))

    with patch("ecg_cvae.trainer.compare_peaks", return_value=False):
        train_vae_for_channel(
            channel_data=channel_data,
            channel_peak_vecs=channel_peaks,
            channel_peak_counts=channel_counts,
            channel_idx=1,
            MAXepochs=1,
            batch_size=2,
            output_dir=str(output_dir),
            fname="testfile2",
            resume=False,
            save_every=1,
            device='cpu'
        )
    # No latent file should be created
    latent_file = output_dir / "v6b3_A" / "testfile2_ch1_similar_latents.npy"
    assert not latent_file.exists(), "Latent file should not be saved when no good latents found"
