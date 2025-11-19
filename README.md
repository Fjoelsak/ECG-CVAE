# ECG-CVAE

## Project structure

```
ECG_CVAE/
│
├── data/                   # Raw data & optionally stored .npy files
├── output/                 # Results: plots, checkpoints, generated samples
│
├── ecg_vae/                # Main module
│   ├── __init__.py
│   ├── preprocessing.py    # Signal filtering, normalization, smoothing, peak detection
│   ├── dataset.py          # Dataset classes
│   ├── models.py           # VAE encoder/decoder, Conditional VAE
│   ├── losses.py           # Morphology loss, VAE loss
│   ├── trainer.py          # Training loop, checkpointing, evaluation
│   ├── sampling.py         # Mahalanobis sampling, latent space sampling
│   ├── plotting.py         # Plot functions for reconstructions and generated samples
│   └── utils.py            # Utility functions: normalization, file loading, peak comparison, etc.
│
├── scripts/                # CLI scripts / experiments
│   ├── train_channel.py    # Training a single channel
│   ├── evaluate.py         # Model evaluation
│   └── generate_samples.py # Generation of samples from latent space
│
├── tests/                  # Tests
│
├── requirements.txt
└── README.md
```