import numpy as np
import torch
import pytest

from ecg_cvae.dataset import ChannelDataset


def test_dataset_shapes():
    N, L = 5, 100
    channel = np.random.randn(N, L)
    peaks = np.random.randint(0, 2, size=(N, L))
    counts = np.arange(N)

    ds = ChannelDataset(channel, peaks, counts)

    assert len(ds) == N

    s, p, c = ds[0]
    assert s.shape == (1, L)
    assert p.shape == (1, L)
    assert c.shape == (1,)


def test_peak_count_normalization():
    N, L = 4, 20
    channel = np.random.randn(N, L)
    peaks = np.zeros((N, L))
    counts = np.array([0, 5, 10, 15])

    ds = ChannelDataset(channel, peaks, counts)

    # normalized: / 15
    expected = counts / 15.0

    for i in range(N):
        _, _, c = ds[i]
        assert torch.isclose(c[0], torch.tensor(expected[i], dtype=torch.float32))


def test_no_peaks_case():
    """If all peak_counts = 0, normalization must NOT divide by zero."""
    N, L = 3, 50
    channel = np.random.randn(N, L)
    peaks = np.zeros((N, L))
    counts = np.zeros(N)

    ds = ChannelDataset(channel, peaks, counts)

    for i in range(N):
        _, _, c = ds[i]
        assert c.item() == 0.0  # still zero, no division


def test_invalid_channel_dimension():
    """Channel data must be 2D; raise AssertionError otherwise."""
    channel = np.random.randn(10)  # wrong shape
    peaks = np.random.randint(0, 2, size=(10, 10))
    counts = np.zeros(10)

    with pytest.raises(AssertionError):
        ChannelDataset(channel, peaks, counts)


def test_item_returns_independent_tensors():
    """Ensure __getitem__ returns separate tensors, not references."""
    N, L = 4, 20
    channel = np.random.randn(N, L)
    peaks = np.random.randint(0, 2, size=(N, L))
    counts = np.arange(N)  # different values to be safer

    ds = ChannelDataset(channel, peaks, counts)

    s1, p1, c1 = ds[0]
    s2, p2, c2 = ds[1]

    # Check memory pointers, ensures tensors are separate
    assert s1.data_ptr() != s2.data_ptr()
    assert p1.data_ptr() != p2.data_ptr()
    assert c1.data_ptr() != c2.data_ptr()

