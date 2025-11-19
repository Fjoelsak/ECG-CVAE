import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt, savgol_filter, find_peaks


def signal_filter(x, fs=500.0, hp_cutoff=0.5, lp_cutoff=40.0, notch_freq=50.0, notch_Q=30.0):
    """
    Apply standard ECG preprocessing filters:
    high-pass, low-pass, notch filtering and baseline wander removal.

    This function implements a commonly used ECG filtering pipeline:
    - 0.5 Hz high-pass to remove baseline drift
    - 40 Hz low-pass to remove high-frequency noise
    - 50 Hz notch to remove powerline interference
    - Median filter-based baseline subtraction (≈200 ms window)

    Parameters
    ----------
    x : array_like
        1D ECG signal.
    fs : float, optional
        Sampling frequency in Hz. Default is 500 Hz.
    hp_cutoff : float, optional
        High-pass filter cutoff frequency in Hz.
    lp_cutoff : float, optional
        Low-pass filter cutoff frequency in Hz.
    notch_freq : float, optional
        Frequency for the notch filter (typically 50 or 60 Hz).
    notch_Q : float, optional
        Quality factor for the notch filter.

    Returns
    -------
    x_corr : ndarray
        Filtered and baseline-corrected ECG signal.

    Notes
    -----
    The baseline wander is removed using a median filter with a window of
    approximately 200 ms. This is a standard approach in ECG pipelines.
    """
    b_hp, a_hp = butter(2, hp_cutoff / (fs/2), btype='highpass')
    x_hp = filtfilt(b_hp, a_hp, x)
    b_lp, a_lp = butter(2, lp_cutoff / (fs/2), btype='lowpass')
    x_band = filtfilt(b_lp, a_lp, x_hp)
    b_notch, a_notch = iirnotch(notch_freq/(fs/2), notch_Q)
    x_notched = filtfilt(b_notch, a_notch, x_band)
    baseline = medfilt(x_notched, kernel_size=int(fs*0.2)+1)
    x_corr = x_notched - baseline
    return x_corr


def ecg_smoothen_npoint(x, N=5, method='mean'):
    """
    Apply N-point moving mean or median smoothing to ECG data.

    Supports 1D, 2D, and 3D inputs:
    - 1D:          (samples,)
    - 2D:          (batch, samples)
    - 3D (ECG):    (batch, channels, samples)

    Parameters
    ----------
    x : array_like
        ECG signal or batch of ECG signals.
    N : int, optional
        Window size of the smoothing kernel.
    method : {'mean', 'median'}, optional
        Smoothing method to apply.

    Returns
    -------
    out : ndarray
        Smoothed signal(s), same shape as input.

    Raises
    ------
    ValueError
        If input dimensionality exceeds 3D or unsupported method is given.

    Notes
    -----
    This function is intended as an optional smoothing step after filtering.
    """
    x_arr = np.asarray(x)
    if x_arr.ndim == 1:
        return _smooth_1d(x_arr, N, method)
    if x_arr.ndim == 2:
        out = np.empty_like(x_arr, dtype=float)
        for i in range(x_arr.shape[0]):
            out[i, :] = _smooth_1d(x_arr[i, :], N, method)
        return out
    if x_arr.ndim == 3:
        out = np.empty_like(x_arr, dtype=float)
        for i in range(x_arr.shape[0]):
            for c in range(x_arr.shape[1]):
                out[i, c, :] = _smooth_1d(x_arr[i, c, :], N, method)
        return out
    raise ValueError("Input must be 1D/2D/3D")


def _smooth_1d(sig, N, method):
    """
    Internal helper to smooth a 1D signal using mean or median filter.

    Parameters
    ----------
    sig : array_like
        1D input signal.
    N : int
        Kernel/window size.
    method : {'mean', 'median'}
        Type of smoothing filter.

    Returns
    -------
    filtered : ndarray
        Smoothed signal.

    Raises
    ------
    ValueError
        If `method` is not supported.

    Notes
    -----
    Median smoothing automatically adjusts to an odd window size.
    Mean smoothing uses convolution.
    """
    if method == 'mean':
        kernel = np.ones(N) / N
        return np.convolve(sig, kernel, mode='same')
    elif method == 'median':
        k = N if N % 2 == 1 else N + 1
        return medfilt(sig, kernel_size=k)
    else:
        raise ValueError("method must be 'mean' or 'median'")


def ecg_smoothen_savgol(x, window_length=15, polyorder=4):
    """
    Apply Savitzky–Golay smoothing to a batch of ECG signals.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_points)
        2D array of ECG signals.
    window_length : int, optional
        Length of the filter window; must be odd.
        If even, it is incremented automatically.
        If larger than signal length, it is reduced.
    polyorder : int, optional
        Order of the polynomial used to fit the samples.

    Returns
    -------
    x_smoothed : ndarray
        Smoothed ECG signals with same shape as input.

    Notes
    -----
    Savitzky–Golay filtering preserves morphology better than moving averages.
    """
    examples, samples = x.shape
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, samples if samples % 2 == 1 else samples - 1)
    x_smoothed = np.empty_like(x, dtype=float)
    for i in range(examples):
        x_smoothed[i, :] = savgol_filter(x[i, :], window_length=window_length, polyorder=polyorder, mode='interp')
    return x_smoothed


def signal_normalize(x):
    """
    Normalize ECG range to [-1, 1].

    Parameters
    ----------
    x : array_like
        Input ECG data of arbitrary shape.

    Returns
    -------
    x_norm : ndarray
        Normalized ECG data with same shape as input.

    Notes
    -----
    If the signal is constant (zero dynamic range), a zero array is returned.
    """
    amin, amax = x.min(), x.max()
    if amax - amin != 0:
        return 2 * ((x - amin) / (amax - amin)) - 1
    else:
        return np.zeros_like(x)


    # ------------------ Peak detection & conditioning ------------------
def detect_peaks_for_dataset(data, fs=500, min_distance=200, prominence=None):
    """
    Detect R-peak–like local maxima for a batched multi-channel ECG dataset.

    Parameters
    ----------
    data : np.ndarray
        Input ECG data of shape (samples, channels, length).
        Each entry data[i, ch, :] contains one ECG trace.
    fs : int or float, optional (default=500)
        Sampling frequency in Hz.
    min_distance : int or float, optional (default=200)
        Minimum distance between two peaks **in milliseconds**.
        Converted internally to samples = min_distance/1000 * fs.
    prominence : float or None, optional (default=None)
        Optional prominence threshold passed to `scipy.signal.find_peaks`.
        If None, a heuristic prominence is computed individually for each signal:
            prom = 0.3 * (max(signal) - median(signal))

    Returns
    -------
    peak_vecs : np.ndarray
        Binary array of shape (samples, channels, length) with 1.0 at detected
        peak positions and 0.0 elsewhere.
    peak_counts : np.ndarray
        Integer array of shape (samples, channels) containing the number of
        detected peaks per (sample, channel).

    Notes
    -----
    - Peak detection makes use of `scipy.signal.find_peaks`.
    - Intended for typical ECG R-peak estimation before segmentation,
      morphology comparison, or conditioning a VAE input.
    - Works on arbitrary batch sizes and multi-channel ECGs.

    Examples
    --------
    >>> data = np.random.randn(10, 2, 1000)
    >>> peaks, counts = detect_peaks_for_dataset(data, fs=500)
    >>> peaks.shape
    (10, 2, 1000)
    >>> counts.shape
    (10, 2)
    """
    samples, channels, length = data.shape
    peak_vecs = np.zeros_like(data, dtype=np.float32)
    peak_counts = np.zeros((samples, channels), dtype=np.int32)
    for i in range(samples):
        for ch in range(channels):
            sig = data[i, ch, :]
            distance = int(min_distance / 1000.0 * fs)  # convert ms to samples
            # heuristic for a prominance value
            if prominence is None:
                heur = 0.3 * (np.max(sig) - np.median(sig))
                prom = heur if heur > 0 else None
            else:
                prom = prominence
            # scipy method identifying local maxima typically corresponding to R-waves
            peaks, props = find_peaks(sig, distance=distance, prominence=prom)
            peak_vecs[i, ch, peaks] = 1.0  # binary vector indicating peak locations
            peak_counts[i, ch] = len(peaks)
    return peak_vecs, peak_counts
