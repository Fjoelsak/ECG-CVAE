# test_sampling.py
import unittest
import numpy as np
from ecg_cvae.sampling import mahalanobis_latent_sampling

class TestMahalanobisLatentSampling(unittest.TestCase):

    def test_output_shape(self):
        """Test that output has the correct shape."""
        latents = np.random.randn(50, 10)
        z = mahalanobis_latent_sampling(latents, latent_dim=10, n_samples=20, random_state=42)
        self.assertEqual(z.shape, (20, 10))

    def test_insufficient_latents(self):
        """Test behavior when latents are None or too few."""
        z = mahalanobis_latent_sampling(None, latent_dim=5, n_samples=10, random_state=42)
        self.assertEqual(z.shape, (10, 5))

        z = mahalanobis_latent_sampling(np.array([[0, 1]]), latent_dim=2, n_samples=5, random_state=42)
        self.assertEqual(z.shape, (5, 2))

    def test_mahalanobis_filtering(self):
        """Test Mahalanobis filtering returns the requested number of samples."""
        latents = np.random.randn(100, 4)
        z = mahalanobis_latent_sampling(latents, latent_dim=4, n_samples=50, max_mahal=0.1, random_state=123)
        self.assertEqual(z.shape, (50, 4))

    def test_random_state_reproducibility(self):
        """Test that using the same random_state produces reproducible results."""
        latents = np.random.randn(30, 3)
        z1 = mahalanobis_latent_sampling(latents, latent_dim=3, n_samples=10, random_state=7)
        z2 = mahalanobis_latent_sampling(latents, latent_dim=3, n_samples=10, random_state=7)
        np.testing.assert_array_almost_equal(z1, z2)

if __name__ == "__main__":
    unittest.main()
