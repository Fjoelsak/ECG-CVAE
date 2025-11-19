import numpy as np
from tkinter import Tk, filedialog
import os
from ecg_cvae.preprocessing import signal_filter, signal_normalize

def load_data_and_preprocess(data, fname, apply_filter=False):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
    if not file_path:
        raise RuntimeError("No file selected.")
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