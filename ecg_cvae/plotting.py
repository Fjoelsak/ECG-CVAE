import os
import matplotlib.pyplot as plt
import numpy as np

# ------------------ Training / plotting / checkpointing ------------------
def plot_generated_samples(samples, channel_idx, output_dir, hidden_dim, latent_dim, epoch, fname=None, latents=None):
    """
    Plot generated samples from a model and save the figure to disk.

    Parameters
    ----------
    samples : list or np.ndarray
        List or array of generated signals, each of shape (signal_length,).
    channel_idx : int
        Index of the channel for labeling purposes.
    output_dir : str
        Directory where the plot image will be saved.
    hidden_dim : int
        Hidden dimension of the model, included in filename.
    latent_dim : int
        Latent dimension of the model, included in filename.
    epoch : int
        Current training epoch, included in filename.
    fname : str, optional
        Base filename to use for saving the plot. Default is None.
    latents : np.ndarray, optional
        Optional latent vectors corresponding to samples, used for annotating titles.
        Only the first 6 values of each latent vector are shown.

    Notes
    -----
    Creates one subplot per sample. If `latents` are provided, each subplot
    title will include the first six values of the corresponding latent vector.
    The figure is saved as a PNG and closed to free memory.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].plot(samples[i])
        if latents is not None:
            latent_info = np.array2string(latents[i][:6], precision=2, separator=',')
            axes[i].set_title(f"Sample {i+1} | z={latent_info}")
        axes[i].axis('off')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{fname}_hdn{hidden_dim}_ltnt_{latent_dim}_ch{channel_idx}_ep{epoch}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[Channel {channel_idx}] Sample plot saved: {plot_path}")


def plot_reconstructions(orig, recon, channel_idx, out_dir, hidden_dim, latent_dim, epoch, fname, good_flags=None):
    """
    Plot original vs reconstructed signals for visual inspection.

    Parameters
    ----------
    orig : np.ndarray or torch.Tensor
        Original signals, shape (num_samples, 1, length) or (num_samples, length).
    recon : np.ndarray or torch.Tensor
        Reconstructed signals, same shape as `orig`.
    channel_idx : int
        Index of the channel for labeling.
    out_dir : str
        Directory where the plot image will be saved.
    hidden_dim : int
        Hidden dimension of the model, included in filename.
    latent_dim : int
        Latent dimension of the model, included in filename.
    epoch : int
        Current training epoch, included in filename.
    fname : str
        Base filename to use for saving the plot.
    good_flags : list of bool, optional
        Optional flags marking "good" reconstructions; flagged samples are annotated with '*'.

    Notes
    -----
    Plots one subplot per sample. The original signal is plotted with a transparent blue line,
    while the reconstruction is overlaid in black. If `good_flags` is provided, flagged
    reconstructions will have a '*' in the legend.
    The figure is saved as a PNG and closed to free memory.
    """
    os.makedirs(out_dir, exist_ok=True)
    num = orig.shape[0]

    fig, axes = plt.subplots(num, 1, figsize=(8, 2 * num))
    if num == 1:
        axes = [axes]

    for i in range(num):
        label_suffix = " *" if (good_flags is not None and i < len(good_flags) and good_flags[i]) else ""
        axes[i].plot(orig[i, 0, :] if orig.ndim == 3 else orig[i, :],
                     label=f"orig", color='blue', alpha=0.3)
        axes[i].plot(recon[i, 0, :] if recon.ndim == 3 else recon[i, :],
                     label=f"recon{label_suffix}", color='black', linestyle='-', alpha=0.7)
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].set_title(f"Channel {channel_idx} | Sample {i+1}")

    plt.tight_layout()
    out_path = os.path.join(
        out_dir,
        f"{fname}_ch{channel_idx}_epoch{epoch}_hd{hidden_dim}_lt{latent_dim}.png"
    )
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Channel {channel_idx}] Saved reconstruction plot to {out_path}")
