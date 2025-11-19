import numpy as np
from scipy.signal import find_peaks
import torch
import torch.nn.functional as F

def morphology_loss(x: torch.Tensor, recon_x: torch.Tensor, prominence_threshold: float = 0.2) -> torch.Tensor:
    """
    Computes a morphological loss between the original and reconstructed ECG signals.

    This loss penalizes differences in:
      - Number of detected peaks
      - Average peak prominence

    Args:
        x (torch.Tensor): Original signal tensor of shape (batch, 1, seq_len)
        recon_x (torch.Tensor): Reconstructed signal tensor of shape (batch, 1, seq_len)
        prominence_threshold (float): Minimum peak prominence used by scipy's find_peaks

    Returns:
        torch.Tensor: Average morphological difference per batch
    """
    batch_size = x.size(0)
    total_peak_diff = 0.0
    total_prominence_diff = 0.0

    for i in range(batch_size):
        signal = x[i, 0].cpu().numpy()
        recon_signal = recon_x[i, 0].detach().cpu().numpy()

        peaks_orig, props_orig = find_peaks(signal, prominence=prominence_threshold)
        peaks_recon, props_recon = find_peaks(recon_signal, prominence=prominence_threshold)

        peak_count_diff = abs(len(peaks_orig) - len(peaks_recon))

        prom_orig = np.mean(props_orig["prominences"]) if "prominences" in props_orig and len(props_orig["prominences"]) > 0 else 0.0
        prom_recon = np.mean(props_recon["prominences"]) if "prominences" in props_recon and len(props_recon["prominences"]) > 0 else 0.0

        prominence_diff = abs(prom_orig - prom_recon)

        total_peak_diff += peak_count_diff
        total_prominence_diff += prominence_diff

    return (total_peak_diff + total_prominence_diff) / float(batch_size)


def loss_fn(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, weight_morph: float = 1.0):
    """
    Combined VAE loss with reconstruction, KL divergence, and morphological penalty.

    Args:
        recon_x (torch.Tensor): Reconstructed signal (batch, 1, seq_len)
        x (torch.Tensor): Original signal (batch, 1, seq_len)
        mu (torch.Tensor): Mean from VAE encoder (batch, latent_dim)
        logvar (torch.Tensor): Log-variance from VAE encoder (batch, latent_dim)
        weight_morph (float): Weight for the morphological loss

    Returns:
        tuple: (total_loss, recon_loss, kld_loss, morph_loss)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Morphological loss
    morph_loss = morphology_loss(x, recon_x)
    morph_loss_t = torch.tensor(morph_loss, device=x.device, dtype=torch.float32)

    # Total loss
    total_loss = recon_loss + kld + weight_morph * morph_loss_t
    return total_loss, recon_loss, kld, morph_loss_t
