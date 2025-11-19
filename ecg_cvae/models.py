import torch
import torch.nn.functional as F
from torch import nn

# ------------------ Conditional VAE (encoder + decoder) ------------------
class CondEncoder(nn.Module):
    """
    1D Convolutional Encoder for Conditional VAE.

    Args:
        in_ch (int): Number of input channels (signal + conditioning channels).
        hidden_dim (int): Hidden dimension size (unused directly but for reference).
        latent_dim (int): Dimension of latent vector z.
        seq_len (int): Length of input sequences.
        kernel_size (int): Kernel size for Conv1d layers.

    Forward:
        x: torch.Tensor of shape (batch, in_ch, seq_len)

    Returns:
        mu: torch.Tensor of shape (batch, latent_dim)
        logvar: torch.Tensor of shape (batch, latent_dim)
    """
    def __init__(self, in_ch, hidden_dim, latent_dim, seq_len=5000, kernel_size=4):
        """
        in_ch: input channels (1 + cond_channels) e.g., 1 + 1 (peak_vec)
        We'll compute flattened size dynamically to avoid matmul mismatch.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 20, kernel_size=kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv1d(20, 40, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv1d(40, 128, kernel_size=kernel_size, stride=2, padding=1)
        self.dropout = nn.Dropout(0.2)
        # dynamic flattened size:
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, seq_len)
            h = self.conv1(dummy)
            h = self.conv2(h)
            h = self.conv3(h)
            flat_size = h.view(1, -1).shape[1]
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self._flat_size = flat_size

    def forward(self, x):
        # x: (batch, in_ch, seq_len)
        x = F.relu(self.conv1(x))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class CondDecoder(nn.Module):
    """
    1D Convolutional Decoder for Conditional VAE.

    Args:
        out_ch (int): Number of output channels (usually 1 for ECG signal).
        hidden_dim (int): Hidden dimension size (unused directly but for reference).
        latent_dim (int): Dimension of latent vector z.
        cond_embed_dim (int): Dimension of conditioning embedding concatenated to z.
        seq_len (int): Length of output sequences.
        kernel_size (int): Kernel size for ConvTranspose1d layers.

    Forward:
        z: torch.Tensor of shape (batch, latent_dim)
        cond_embed: torch.Tensor of shape (batch, cond_embed_dim)

    Returns:
        recon: torch.Tensor of shape (batch, out_ch, seq_len)
    """
    def __init__(self, out_ch, hidden_dim, latent_dim, cond_embed_dim, seq_len=5000, kernel_size=4):
        """
        Decoder receives z concatenated with cond_embed (scalar embedding from peak count or pooled cond).
        cond_embed_dim: dimension of the conditioning vector concatenated to z.
        """
        super().__init__()
        self.latent_dim = latent_dim                 # for external reference / logging
        self.cond_embed_dim = cond_embed_dim         # explicit reference
        self.latent_in = latent_dim + cond_embed_dim # total input dimension to fc
        # The decoder will map to same 128 * 625 (or derived) shape. To be safe, we compute init_len from encoder downsample ratio.
        # We pick init_len = seq_len // 8 // 2 // 2 ... according to downsampling of encoder (conv stride=2 x3 -> /8)
        # But we will compute an init_len that matches the encoder used above if needed; choose 625 as in original if seq_len==5000
        self.init_len = seq_len // (2 * 2 * 2)  # seq_len // 8; for 5000 -> 625 (approx) since each stride=2 and 3 layers.
        if self.init_len < 1:
            self.init_len = 1
        self.fc = nn.Linear(self.latent_in, 128 * self.init_len)
        self.deconv1 = nn.ConvTranspose1d(128, 40, kernel_size=kernel_size, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(40, 20, kernel_size=kernel_size, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(20, out_ch, kernel_size=kernel_size, stride=2, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.seq_len = seq_len

    def forward(self, z, cond_embed):
        # z: (batch, latent_dim), cond_embed: (batch, cond_embed_dim)
        x = torch.cat([z, cond_embed], dim=1)
        x = self.fc(x).view(-1, 128, self.init_len)
        x = self.dropout(F.relu(self.deconv1(x)))
        x = self.dropout(F.relu(self.deconv2(x)))
        out = torch.tanh(self.deconv3(x))
        # Crop/pad to seq_len if necessary
        if out.size(2) > self.seq_len:
            out = out[:, :, :self.seq_len]
        elif out.size(2) < self.seq_len:
            pad = self.seq_len - out.size(2)
            out = F.pad(out, (0, pad))
        return out

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for 1D signals (e.g., ECG) with peak conditioning.

    Args:
        seq_len (int): Sequence length of input signals.
        hidden_dim (int): Hidden dimension reference for encoder/decoder.
        latent_dim (int): Dimension of latent vector z.
        cond_channels (int): Number of conditioning channels concatenated to input (e.g., 1 for peak vector).
        cond_embed_dim (int): Dimension of embedding of conditioning vector concatenated to latent vector.

    Forward:
        x: torch.Tensor of shape (batch, 1, seq_len) - input signal
        peak_vec: torch.Tensor of shape (batch, 1, seq_len) - binary peak vector
        peak_count: torch.Tensor of shape (batch, 1) - scalar normalized peak count

    Returns:
        recon: torch.Tensor of shape (batch, 1, seq_len) - reconstructed signal
        mu: torch.Tensor of shape (batch, latent_dim) - mean of latent distribution
        logvar: torch.Tensor of shape (batch, latent_dim) - log-variance of latent distribution
        z: torch.Tensor of shape (batch, latent_dim) - sampled latent vector
    """
    def __init__(self, seq_len=5000, hidden_dim=128, latent_dim=128, cond_channels=1, cond_embed_dim=8):
        """
        cond_channels: number of conditioning channels concatenated to input (e.g., 1 for peak vector)
        cond_embed_dim: dimension of the conditioning embedding concatenated to latent vector (we'll use peak count + pooled cond -> small vector)
        """
        super().__init__()
        self.seq_len = seq_len
        self.cond_channels = cond_channels
        in_ch = 1 + cond_channels  # signal + binary peak vector (as channels)
        self.encoder = CondEncoder(in_ch=in_ch, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len)
        # cond_embed_dim + latent_dim -> decoder input
        self.decoder = CondDecoder(out_ch=1, hidden_dim=hidden_dim, latent_dim=latent_dim, cond_embed_dim=cond_embed_dim, seq_len=seq_len)
        # small MLP to embed conditioning (peak vector + count) into cond_embed_dim
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_channels + 1, 32),
            nn.ReLU(),
            nn.Linear(32, cond_embed_dim),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, peak_vec, peak_count):
        """
        x: (batch,1,L) signal
        peak_vec: (batch,1,L) binary vector
        peak_count: (batch,1) scalar normalized [0,1]
        """
        # concat on channel dim for encoder
        enc_in = torch.cat([x, peak_vec], dim=1)  # (batch,1+cond_ch,L)
        mu, logvar = self.encoder(enc_in)
        z = self.reparameterize(mu, logvar)
        # produce cond summary: pool peak_vec along time (mean) to get per-sample value(s)
        pooled = peak_vec.mean(dim=2)  # (batch, 1)
        cond_cat = torch.cat([pooled, peak_count], dim=1)  # (batch, 2)
        cond_embed = self.cond_embedding(cond_cat)  # (batch, cond_embed_dim)
        recon = self.decoder(z, cond_embed)
        return recon, mu, logvar, z