import numpy as np
import torch
from torch.utils.data import Dataset

class ChannelDataset(Dataset):
    """
    PyTorch dataset for ECG channel data including the raw signal,
    peak indicator vectors, and normalized peak counts.

    Parameters
    ----------
    channel_data : np.ndarray
        Array of shape (N, L) containing ECG samples for one channel.
        N = number of samples, L = sequence length.
    peak_vecs : np.ndarray
        Binary array of shape (N, L) indicating peak locations.
    peak_counts : np.ndarray
        Array of shape (N,) containing the number of peaks per sample
        for the corresponding channel.

    Notes
    -----
    - The data are converted to `torch.float32`.
    - `peak_counts` is normalized to the range [0, 1] by dividing by `max(counts)`,
      unless all counts are zero.
    """

    def __init__(self, channel_data, peak_vecs, peak_counts):
        assert channel_data.ndim == 2, "channel_data must be a 2D array (N, L)"

        self.signals = torch.tensor(channel_data, dtype=torch.float32).unsqueeze(1)
        self.peak_vecs = torch.tensor(peak_vecs, dtype=torch.float32).unsqueeze(1)

        counts = peak_counts.astype(np.float32)
        if counts.max() > 0:
            counts_norm = counts / counts.max()
        else:
            counts_norm = counts

        self.counts = torch.tensor(counts_norm.reshape(-1, 1), dtype=torch.float32)
        self.length = channel_data.shape[1]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.signals.shape[0]

    def __getitem__(self, idx):
        """
        Get the dataset sample at index `idx`.

        Returns
        -------
        signal : torch.Tensor
            Tensor of shape (1, L) containing the ECG signal.
        peak_vec : torch.Tensor
            Tensor of shape (1, L) containing the binary peak locations.
        count : torch.Tensor
            Tensor of shape (1,) containing the normalized peak count.
        """
        return self.signals[idx], self.peak_vecs[idx], self.counts[idx]
