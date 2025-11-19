import torch
import pytest
from ecg_cvae.models import CondEncoder, CondDecoder, ConditionalVAE

def test_cond_encoder_output_shapes():
    batch, seq_len, in_ch, latent_dim = 4, 1000, 2, 16
    x = torch.randn(batch, in_ch, seq_len)
    encoder = CondEncoder(in_ch=in_ch, hidden_dim=32, latent_dim=latent_dim, seq_len=seq_len)
    mu, logvar = encoder(x)
    assert mu.shape == (batch, latent_dim)
    assert logvar.shape == (batch, latent_dim)

def test_cond_decoder_output_shape():
    batch, seq_len, latent_dim, cond_embed_dim = 4, 1000, 16, 8
    z = torch.randn(batch, latent_dim)
    cond_embed = torch.randn(batch, cond_embed_dim)
    decoder = CondDecoder(out_ch=1, hidden_dim=32, latent_dim=latent_dim,
                          cond_embed_dim=cond_embed_dim, seq_len=seq_len)
    recon = decoder(z, cond_embed)
    assert recon.shape == (batch, 1, seq_len)

def test_conditional_vae_forward():
    batch, seq_len = 4, 1000
    latent_dim, cond_channels, cond_embed_dim = 16, 1, 8
    x = torch.randn(batch, 1, seq_len)
    peak_vec = torch.randint(0, 2, (batch, cond_channels, seq_len), dtype=torch.float32)
    peak_count = torch.rand(batch, 1)

    model = ConditionalVAE(seq_len=seq_len, hidden_dim=32, latent_dim=latent_dim,
                           cond_channels=cond_channels, cond_embed_dim=cond_embed_dim)
    recon, mu, logvar, z = model(x, peak_vec, peak_count)

    assert recon.shape == (batch, 1, seq_len)
    assert mu.shape == (batch, latent_dim)
    assert logvar.shape == (batch, latent_dim)
    assert z.shape == (batch, latent_dim)
