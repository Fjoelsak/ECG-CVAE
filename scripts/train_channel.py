import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from ecg_cvae.preprocessing import detect_peaks_for_dataset
from ecg_cvae.utils import load_data_and_preprocess
from ecg_cvae.trainer import train_vae_for_channel


# ------------------ Main ------------------
def main(load_mode="local", hidden_dim=128, latent_dim=128, plot_training_examples=True):
    device = 'cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu'

    print("Select the .npy file with shape (samples, channels, length).")
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        filetypes=[("Numpy files", "*.npy")]
    )

    if not file_path:
        raise RuntimeError("No file selected.")

    data = np.load(file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0]

    data, fname = load_data_and_preprocess(data, filename, apply_filter=False)  # file selection
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
            plt.savefig(os.path.join("output", "v6b3_A", f"{filename}_ch{ch}_train_examples.png"))
            plt.close()


if __name__ == "__main__":
    main(hidden_dim=128, latent_dim=24, plot_training_examples=True)
