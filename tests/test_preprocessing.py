import numpy as np
import pytest
from scipy.signal import medfilt

# Modul importieren
from ecg_cvae.preprocessing import (
    signal_filter,
    ecg_smoothen_npoint,
    _smooth_1d,
    ecg_smoothen_savgol,
    signal_normalize,
)

# -----------------------------------------------------
# Helper Test Data
# -----------------------------------------------------

@pytest.fixture
def simple_signal():
    """Provide a simple synthetic 1D ECG-like test signal."""
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
    return signal


@pytest.fixture
def batch_signal():
    """Provide a batch of 10 synthetic ECG signals."""
    t = np.linspace(0, 1, 500)
    return np.array([np.sin(2 * np.pi * 5 * t) for _ in range(10)])


# -----------------------------------------------------
# signal_normalize
# -----------------------------------------------------

def test_signal_normalize_range():
    x = np.array([0, 1, 2, 3, 4])
    norm = signal_normalize(x)
    assert np.isclose(norm.min(), -1.0)
    assert np.isclose(norm.max(), 1.0)


def test_signal_normalize_constant():
    x = np.ones(100)
    norm = signal_normalize(x)
    assert np.allclose(norm, np.zeros_like(x))


# -----------------------------------------------------
# _smooth_1d
# -----------------------------------------------------

def test_smooth_1d_mean():
    x = np.array([1, 2, 3, 4, 5])
    smoothed = _smooth_1d(x, N=3, method='mean')
    assert smoothed.shape == x.shape
    assert not np.isnan(smoothed).any()


def test_smooth_1d_median():
    x = np.array([1, 100, 3, 4, 5])
    smoothed = _smooth_1d(x, N=3, method='median')
    assert smoothed.shape == x.shape
    assert smoothed[1] == np.median([1, 100, 3])


def test_smooth_1d_invalid_method():
    x = np.arange(10)
    with pytest.raises(ValueError):
        _smooth_1d(x, N=5, method='invalid')


# -----------------------------------------------------
# ecg_smoothen_npoint
# -----------------------------------------------------

def test_ecg_smoothen_npoint_1d(simple_signal):
    sm = ecg_smoothen_npoint(simple_signal, N=5, method='mean')
    assert sm.shape == simple_signal.shape


def test_ecg_smoothen_npoint_2d(batch_signal):
    sm = ecg_smoothen_npoint(batch_signal, N=5, method='median')
    assert sm.shape == batch_signal.shape


def test_ecg_smoothen_npoint_3d():
    x = np.random.randn(4, 2, 500)
    sm = ecg_smoothen_npoint(x, N=5, method='mean')
    assert sm.shape == x.shape


def test_ecg_smoothen_npoint_invalid_dim():
    x = np.random.randn(2, 2, 2, 500)
    with pytest.raises(ValueError):
        ecg_smoothen_npoint(x, N=5, method='mean')


# -----------------------------------------------------
# ecg_smoothen_savgol
# -----------------------------------------------------

def test_ecg_smoothen_savgol_shape(batch_signal):
    sm = ecg_smoothen_savgol(batch_signal, window_length=15, polyorder=3)
    assert sm.shape == batch_signal.shape


def test_ecg_smoothen_savgol_odd_window():
    x = np.random.randn(3, 100)
    sm = ecg_smoothen_savgol(x, window_length=10, polyorder=3)
    assert sm.shape == x.shape


def test_ecg_smoothen_savgol_reduces_noise():
    np.random.seed(0)
    clean = np.sin(np.linspace(0, 10, 500))
    noisy = clean + 0.5 * np.random.randn(500)
    sm = ecg_smoothen_savgol(noisy.reshape(1, -1), window_length=21, polyorder=3)[0]

    # noise should be reduced (std smaller)
    assert sm.std() < noisy.std()


# -----------------------------------------------------
# signal_filter
# -----------------------------------------------------

def test_signal_filter_runs(simple_signal):
    filtered = signal_filter(simple_signal.copy(), fs=500)
    assert filtered.shape == simple_signal.shape
    assert not np.isnan(filtered).any()


def test_signal_filter_baseline_removed():
    # Construct artificial baseline drift
    t = np.linspace(0, 10, 5000)
    baseline = 0.5 * np.sin(0.2 * np.pi * t)
    signal = np.sin(2 * np.pi * 5 * t) + baseline

    filtered = signal_filter(signal, fs=500)

    # Baseline should be significantly reduced
    assert filtered.std() < signal.std()
