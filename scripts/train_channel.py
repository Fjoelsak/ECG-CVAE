import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from ecg_cvae.preprocessing import detect_peaks_for_dataset
from ecg_cvae.utils import load_data_and_preprocess
from ecg_cvae.trainer import train_vae_for_channel


def main(data_path=None, hidden_dim=128, latent_dim=24, plot_training_examples=True):
    device = 'cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu'

    if data_path is None:
        data, fname = load_data_and_preprocess(apply_filter=False, file_path=None)
    else:
        # In Colab you will download the file first, then pass its path
        data_path = "downloaded_data.npy"  # example path
        data, fname = load_data_and_preprocess(apply_filter=False, file_path=data_path)

    samples, channels, length = data.shape

    # compute peak vectors & counts for whole dataset
    peak_vecs_all, peak_counts_all = detect_peaks_for_dataset(data, fs=500, min_distance=200, prominence=None)

    # train per-channel
    active_channels = [1]  # or list(range(channels)) for all channels
    for ch in active_channels:
        print(f"\n=== Training channel {ch} ===")
        channel_data = data[:, ch, :]  # (samples, length)
        channel_peak_vecs = peak_vecs_all[:, ch, :]  # (samples, length)
        channel_peak_counts = peak_counts_all[:, ch]  # (samples,)

        train_vae_for_channel(
            channel_data,
            channel_peak_vecs,
            channel_peak_counts,
            ch,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            cond_channels=1,
            cond_embed_dim=8,
            MAXepochs=600,
            batch_size=64,
            lr=1e-3,
            device=device,
            output_dir="output",
            fname=fname,
            resume=True,
            save_every=10,
            weight_morph=1.0
        )

        if plot_training_examples:
            # Plot some training examples for this channel
            os.makedirs(os.path.join("output", "v6b3_A"), exist_ok=True)
            idxs = np.random.choice(len(channel_data), size=min(10, len(channel_data)), replace=False)
            fig, axes = plt.subplots(len(idxs), 1, figsize=(8, 2 * len(idxs)))
            for i, idd in enumerate(idxs):
                axes[i].plot(channel_data[idd])
                axes[i].set_title(f"Chan{ch} Train example {idd}")
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join("output", "v6b3_A", f"{fname}_ch{ch}_train_examples.png"))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train channel-wise CVAE for ECG data."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to .npy dataset file. If omitted, opens Tkinter file dialog.",
    )

    args = parser.parse_args()

    main(data_path=args.data_path, hidden_dim=128, latent_dim=24, plot_training_examples=True)
