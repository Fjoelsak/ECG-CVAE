import numpy as np
import os
from ecg_cvae.preprocessing import signal_filter, signal_normalize

def load_data_and_preprocess(file_path = None, apply_filter=False):
    """
    Load a multi-channel ECG dataset from a `.npy` file and apply normalization
    (and optional filtering) to each sample/channel.

    This function supports two modes:
    - **Interactive mode (local)**: If `file_path` is None, a Tkinter file dialog
      is opened to allow the user to select a `.npy` file.
    - **Direct path mode (e.g., Colab/GitHub asset)**: If `file_path` is provided,
      it is loaded directly without any GUI interaction.

    The input `.npy` file must contain an array of shape:
        `(samples, channels, length)`

    For each sample and channel:
    - If `apply_filter=True`, the signal is passed through `signal_filter()`.
    - The signal is then normalized using `signal_normalize()`.

    The resulting normalized dataset is saved as:
        `NormalizedTrainingData_<original_filename>.npy`

    Parameters
    ----------
    file_path : str or None, optional
        Path to the `.npy` file to load. If None, a Tkinter GUI file selector
        is opened. Default is None.

    apply_filter : bool, optional
        Whether to apply `signal_filter()` before normalization. Default is False.

    Returns
    -------
    normalized : np.ndarray
        The normalized (and optionally filtered) dataset of shape
        `(samples, channels, length)`.

    fileused : str
        The base filename (without extension) of the loaded file, useful
        for naming outputs.

    Raises
    ------
    RuntimeError
        If no file is selected when running in GUI mode.
    """

    if file_path is None:
        print("Select the .npy file with shape (samples, channels, length).")
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if not file_path:
            raise RuntimeError("No file selected.")

    print(f"Loading file: {file_path}")
    path, filename = os.path.split(file_path)
    fileused = os.path.splitext(filename)[0]

    data = np.load(file_path)  # expected shape: (samples, channels, length)

    samples, channels, length = data.shape
    print(f"Loaded {samples} samples, {channels} channels, length {length}")
    normalized = np.empty_like(data, dtype=float)
    print("Normalizing signals (and optionally filtering)...")
    for s in range(samples):
        for c in range(channels):
            x = data[s, c, :]
            if apply_filter:
                x = signal_filter(x)
            normalized[s, c, :] = signal_normalize(x)
    np.save(f'NormalizedTrainingData_{fileused}.npy', normalized)
    print("Saved normalized training data.")
    return normalized, fileused