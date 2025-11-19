import numpy as np

def mahalanobis_latent_sampling(collected_latents, latent_dim, n_samples=100,
                                channel_idx=None, max_mahal=3.0, random_state=None):
    """
    Sample latent vectors from the empirical distribution of collected latents,
    optionally constrained by Mahalanobis distance.

    This function computes the mean and covariance of collected latents and draws
    multivariate Gaussian samples. Samples with a Mahalanobis distance exceeding
    `max_mahal` are optionally filtered out and resampled to maintain the desired
    number of samples.

    Args:
        collected_latents (np.ndarray): Array of shape (N, D) containing previously
            collected latent vectors.
        latent_dim (int): Dimensionality of the latent space (D).
        n_samples (int, optional): Number of latent samples to draw. Defaults to 100.
        channel_idx (int, optional): Index for logging prefix (for debugging). Defaults to None.
        max_mahal (float, optional): Maximum allowed Mahalanobis distance for a sample.
            Samples exceeding this distance are resampled. Defaults to 3.0.
        random_state (int, np.random.Generator, optional): Random seed or RNG instance
            for reproducible sampling. Defaults to None.

    Returns:
        np.ndarray: Array of shape (n_samples, latent_dim) containing sampled latent vectors.
    """
    rng = np.random.default_rng(random_state)
    prefix = f"[Channel {channel_idx}]" if channel_idx is not None else ""

    # Basic sanity check
    if collected_latents is None or collected_latents.shape[0] < 2:
        print(f"{prefix} insufficient latents -> random Gaussian sampling.", flush=True)
        return rng.standard_normal((n_samples, latent_dim))

    # Compute mean and covariance
    mu = collected_latents.mean(axis=0)
    cov = np.cov(collected_latents.T)

    # Regularize covariance for numerical stability
    eps = 1e-6 * np.eye(cov.shape[0])
    cov += eps

    # Cholesky for sampling
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        print(f"{prefix} covariance not PSD, using diagonal covariance.", flush=True)
        std = np.std(collected_latents, axis=0)
        return mu + rng.standard_normal((n_samples, latent_dim)) * std

    # Draw samples
    z_samples = mu + rng.standard_normal((n_samples, latent_dim)) @ L.T

    # Optional: filter by Mahalanobis distance
    cov_inv = np.linalg.inv(cov)
    dM = np.sqrt(np.sum((z_samples - mu) @ cov_inv * (z_samples - mu), axis=1))
    print(f"{prefix} Mahalanobis distances of sampled latents: min={dM.min():.3f}, max={dM.max():.3f}, mean={dM.mean():.3f}", flush=True)
    keep = dM <= max_mahal

    if not np.any(keep):
        print(f"{prefix} all samples rejected by Mahalanobis filter; returning unfiltered Gaussian samples.", flush=True)
        return z_samples
    else:
        accepted = z_samples[keep]
        if accepted.shape[0] < n_samples:
            needed = n_samples - accepted.shape[0]
            print(f"{prefix} accepted {accepted.shape[0]} samples; resampling {needed} more.", flush=True)
            extra = mahalanobis_latent_sampling(collected_latents, latent_dim,
                                                n_samples=needed, channel_idx=None,
                                                max_mahal=max_mahal, random_state=rng)
            accepted = np.vstack([accepted, extra])
        print(f"{prefix} generated {n_samples} latent samples within Mahalanobis distance ≤ {max_mahal}.", flush=True)
        return accepted[:n_samples]
