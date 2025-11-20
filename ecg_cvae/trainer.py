import numpy as np
import torch
import torch.nn.functional as F
import os

from ecg_cvae.dataset import ChannelDataset
from ecg_cvae.losses import loss_fn
from ecg_cvae.models import ConditionalVAE
from ecg_cvae.plotting import plot_generated_samples, plot_reconstructions
from ecg_cvae.sampling import mahalanobis_latent_sampling
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

# ------------------ Evaluation / comparison ------------------
def evaluate_model(model, data, peak_vecs, peak_counts, batch_size=32, device='cpu'):
    """
    Evaluate a trained model on ECG signals using mean squared error (MSE).

    Parameters
    ----------
    model : torch.nn.Module
        Trained ConditionalVAE or compatible model
    data : np.ndarray
        Signal data, shape (samples, length)
    peak_vecs : np.ndarray
        Binary peak indicators, shape (samples, length)
    peak_counts : np.ndarray
        Peak counts per sample, shape (samples,)
    batch_size : int
        Mini-batch size for evaluation
    device : str
        Device to run the model on ('cpu' or 'cuda')

    Returns
    -------
    float
        Average MSE per signal element
    """
    dataset = ChannelDataset(data, peak_vecs, peak_counts)
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for sig, pv, pc in loader:
            sig = sig.to(device)
            pv = pv.to(device)
            pc = pc.to(device)
            recon, _, _, _ = model(sig, pv, pc)
            mse = F.mse_loss(recon, sig, reduction='sum').item()
            total_mse += mse
            count += sig.size(0) * sig.size(-1)
    return total_mse / float(count)

def compare_peaks(original: np.ndarray, reconstructed: np.ndarray,
                  fs: float = 500, min_prominence: float = 0.2,
                  loc_tol: int = 15, height_tol: float = 0.35,
                  min_match_ratio: float = 0.65) -> bool:
    """
    Compares R-peaks between original and reconstructed ECG signals.

    Parameters
    ----------
    original, reconstructed : np.ndarray
        1D ECG waveforms (same length)
    fs : float
        Sampling frequency (Hz)
    min_prominence : float
        Minimum peak prominence for detection
    loc_tol : int
        Tolerance (samples) for matching peak locations
    height_tol : float
        Fractional tolerance for R-peak height difference (e.g., 0.25 = ±25%)
    min_match_ratio : float
        Fraction of peaks that must match to be considered "good"

    Returns
    -------
    bool
        True if reconstruction is morphologically similar (good latent), else False
    """
    peaks_o, props_o = find_peaks(original, prominence=min_prominence)
    peaks_r, props_r = find_peaks(reconstructed, prominence=min_prominence)

    if len(peaks_o) == 0 or len(peaks_r) == 0:
        return False

    if abs(len(peaks_o) - len(peaks_r)) > 0.25 * len(peaks_o):
        return False

    matched = 0
    for i, p in enumerate(peaks_o):
        diffs = np.abs(peaks_r - p)
        if len(diffs) == 0:
            continue
        j = np.argmin(diffs)
        if diffs[j] <= loc_tol:
            h_o = original[p]
            h_r = reconstructed[peaks_r[j]]
            if h_o == 0:
                continue
            height_diff = abs(h_r - h_o) / abs(h_o)
            if height_diff <= height_tol:
                matched += 1

    match_ratio = matched / max(len(peaks_o), 1)
    return match_ratio >= min_match_ratio


# ====================== main training function per channel ======================
def train_vae_for_channel(channel_data, channel_peak_vecs, channel_peak_counts, channel_idx,
                          hidden_dim=256, latent_dim=128, cond_channels=1, cond_embed_dim=8,
                          MAXepochs=200, batch_size=16, lr=1e-3, device='cpu',
                          output_dir="output", fname="datafile", resume=True,
                          save_every=10, weight_morph=1.0):
    """
    Trains a conditional VAE for one channel.
    Collects latent vectors for reconstructions that match original peak structure.
    """
    os.makedirs(os.path.join(output_dir, "v6b3_A"), exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "v6b3_A",
                                   f"{fname}_ch{channel_idx}_hd{hidden_dim}_lt_{latent_dim}_ckpt.pt")
    final_model_path = os.path.join(output_dir, "v6b3_A",
                                    f"{fname}_ch{channel_idx}_hd{hidden_dim}_lt_{latent_dim}_final.pt")
    latent_save_path = os.path.join(output_dir, "v6b3_A",
                                    f"{fname}_ch{channel_idx}_similar_latents.npy")

    dataset = ChannelDataset(channel_data, channel_peak_vecs, channel_peak_counts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConditionalVAE(seq_len=channel_data.shape[1],
                           hidden_dim=hidden_dim,
                           latent_dim=latent_dim,
                           cond_channels=cond_channels,
                           cond_embed_dim=cond_embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Resume if checkpoint exists
    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        start_epoch = ck.get('epoch', 0) + 1
        print(f"[Channel {channel_idx}] Resuming from epoch {start_epoch}")

    # Initialize latent collector
    collected_latents = []
    if resume and os.path.exists(latent_save_path):
        try:
            prev_latents = np.load(latent_save_path)
            collected_latents = prev_latents.tolist()  # convert to list for easy appending
            print(f"[Channel {channel_idx}] Loaded {len(prev_latents)} previously collected latent vectors.")
        except Exception as e:
            print(f"[Channel {channel_idx}] Warning: Failed to load previous latents ({e}). Starting fresh.")

    # ---------------------- Training Loop ----------------------
    for epoch in range(start_epoch, MAXepochs):
        model.train()
        epoch_loss = recon_loss_sum = kld_loss_sum = morph_loss_sum = 0.0

        for sig, pv, pc in loader:
            sig, pv, pc = sig.to(device), pv.to(device), pc.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(sig, pv, pc)

            loss, rec_loss, kld_loss, morph_loss = loss_fn(recon, sig, mu, logvar, weight_morph=weight_morph)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            recon_loss_sum += rec_loss.item()
            kld_loss_sum += kld_loss.item()
            morph_loss_sum += morph_loss.item() if isinstance(morph_loss, torch.Tensor) else float(morph_loss)

        # ---- Epoch summary ----
        n_samples = len(dataset)
        avg_total = epoch_loss / n_samples
        avg_recon = recon_loss_sum / n_samples
        avg_kld = kld_loss_sum / n_samples
        avg_morph = morph_loss_sum / n_samples

        print(f"[Channel {channel_idx}] Epoch {epoch + 1}/{MAXepochs} "
              f"| Total: {avg_total:.5f} | Recon: {avg_recon:.5f} "
              f"| KLD: {avg_kld:.5f} | Morph: {avg_morph:.5f}")

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict()},
                   checkpoint_path)

        # ---- Every few epochs: plot & collect "good" latents ----
        if (epoch + 1) % save_every == 0 or epoch == MAXepochs - 1:
            model.eval()
            good_found = 0
            with torch.no_grad():
                N_GOOD_REQUIRED = 200  # number of good latents to ensure per epoch for large collection.
                MAX_TRIES = int(
                    1000 * (epoch / MAXepochs))  # Progressively increase safety limit (avoid infinite loops)
                print(f"[At Epoch {epoch})] Max_Tries set to {MAX_TRIES}...\n")  # Debugging info (can delete later)
                tries = 0
                origs, recons, good_flags = [], [], []
                print(f"[Channel {channel_idx}] Searching for {N_GOOD_REQUIRED} good latents at epoch {epoch + 1}...")
                while good_found < N_GOOD_REQUIRED and tries < MAX_TRIES:
                    tries += 1
                    idx = np.random.randint(0, len(dataset))
                    s, p, c = dataset[idx]
                    s_b = s.unsqueeze(0).to(device)
                    p_b = p.unsqueeze(0).to(device)
                    c_b = c.unsqueeze(0).to(device)
                    r, mu, logvar, z = model(s_b, p_b, c_b)

                    s_np = s.cpu().numpy().squeeze()
                    r_np = r.cpu().numpy().squeeze()
                    # Check morphology similarity
                    is_good = compare_peaks(s_np, r_np)
                    good_flags.append(is_good)
                    if is_good:
                        collected_latents.append(z.cpu().numpy().squeeze())
                        good_found += 1
                        origs.append(s_np)  # store for plotting only if good enough reconstruction.
                        recons.append(r_np)  # store for plotting only if good enough reconstruction.
                print(f"[Channel {channel_idx}] Found {good_found} good latents after {tries} tries.")

            # ---- Plot the last 10 reconstructions but only if required number of good latents are found----
            # However, there may be less than 10 good ones found; in that case plot whatever is found.
            if good_found >= int(N_GOOD_REQUIRED * 1 / 10):  # plot only if atleast 10% good latents found.
                plot_reconstructions(np.stack(origs[-10:], axis=0),
                                     np.stack(recons[-10:], axis=0), channel_idx,
                                     os.path.join(output_dir, "v6b3_A"), hidden_dim, latent_dim,
                                     epoch + 1, fname, good_flags=good_flags[-10:])

            model.train()

    # ---- Save all collected latents ----
    if len(collected_latents) > 0:
        collected_latents = np.stack(collected_latents, axis=0)
        collected_latents = np.unique(collected_latents, axis=0)  # keep only unique latents
        np.save(latent_save_path, collected_latents)
        print(f"[Channel {channel_idx}] Saved {collected_latents.shape[0]} good latent vectors to {latent_save_path}")
    else:
        print(f"[Channel {channel_idx}] No similar reconstructions found; latent matrix not saved.")
        collected_latents = None

    # ---- Final model save ----
    torch.save({'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'epoch': MAXepochs - 1}, final_model_path)
    print(f"[Channel {channel_idx}] Final model saved: {final_model_path}")

    # ---- Evaluate model ----
    eval_mse = evaluate_model(model, channel_data, channel_peak_vecs,
                              channel_peak_counts, batch_size=batch_size, device=device)
    print(f"[Channel {channel_idx}] Evaluation MSE: {eval_mse:.6f}")

    # ============================================================================================================
    #   Generate final 100 samples from convex hull of "good" latent space i.e. Selective sampling of Latent Space
    # ============================================================================================================
    model.eval()
    with torch.no_grad():
        gen_list, latents_used = [], []
        if collected_latents is not None and collected_latents.shape[0] >= model.decoder.latent_dim + 1:
            print(
                f"[Channel {channel_idx}] Mahalanobis Distance GMM fit using {collected_latents.shape[0]} latent vectors...")
            z_samples = mahalanobis_latent_sampling(collected_latents=collected_latents,
                                                    latent_dim=model.decoder.latent_dim, n_samples=100,
                                                    channel_idx=channel_idx, max_mahal=3)

            # Convert to tensor
        z_tensor = torch.tensor(z_samples, dtype=torch.float32).to(device)

        # Generate conditioning inputs
        gen_dataset_len = len(dataset)
        for i in range(100):
            idx = np.random.randint(0, gen_dataset_len)
            _, pv, pc = dataset[idx]  # select a random sample's conditioning data i.e. peak vector and count
            pv_b = pv.unsqueeze(0).to(device)
            pc_b = pc.unsqueeze(0).to(device)
            pooled = pv_b.mean(dim=2)
            cond_cat = torch.cat([pooled, pc_b], dim=1)
            cond_embed = model.cond_embedding(cond_cat.to(device))

            gen = model.decoder(z_tensor[i].unsqueeze(0), cond_embed)
            gen_list.append(gen.cpu().numpy()[0, 0, :])
            latents_used.append(z_tensor[i].cpu().numpy())

        gen_arr = np.stack(gen_list, axis=0)
        np.save(os.path.join(output_dir, f"{fname}_ch{channel_idx}_generated_100.npy"), gen_arr)
        plot_generated_samples(gen_arr[:10], channel_idx,
                               os.path.join(output_dir, "v6b3_A"),
                               hidden_dim, latent_dim, "final_MHL",
                               fname=fname, latents=np.array(latents_used[:10]))

        print(f"[Channel {channel_idx}] Generated 100 samples using Mahalanobis Distance Gaussian Fit Latents.")
