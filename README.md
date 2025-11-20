# ECG-CVAE

## Link for colab

- train_model.ipynb  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fjoelsak/ECG-CVAE/blob/main/scripts/train_model.ipynb)

## Project structure

```
ECG_CVAE/
│
├── docs/                   # Documentation
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
│   └── utils.py            # Utility functions: file loading
│
├── scripts/                # CLI scripts / experiments
│   └── train_channel.py    # Training a single channel
│
├── tests/                  # Unit Tests
│
├── requirements.txt
└── README.md
```