"""Recompute SNRs from stored whitened signals and compare to the SNRs on
file, for a GWDatasetGeneration combined sig_combined_*.h5.

Adapted from aframe/dev/data/diagnostic/verify_snr.py, but following the
approach already validated in notebooks/snr_segment_check.ipynb: our combined
h5 stores no raw waveform / real background PSD, so the PSD is estimated
per-batch from the stored whitened_bkg via ml4gw's SpectralDensity (NOT
assumed flat/unit — the whitened background isn't perfectly white), then fed
into ml4gw.gw.compute_network_snr on whitened_signal, matching generation's
own compute_network_snr(raw_waveform, real_psd, highpass=f_min) call in
injections.py.

Writes <dest>/snrs.out (per-event stored/recomputed/percent-diff),
<dest>/snr_summary.txt (fraction of events within each percent-diff
threshold), and <dest>/snr_scatter.png.

Example:
    python verify_snr.py \\
        --input /path/to/sig_combined_train.h5 \\
        --config-yaml /path/to/BNS_SNR7-12_TRAIN_VAL_2048Hz_BNS.yaml \\
        --dest /path/to/train/prior_plots
"""

import argparse
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_config
from ml4gw.gw import compute_network_snr
from ml4gw.transforms import SpectralDensity

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="sig_combined_*.h5 file")
    parser.add_argument(
        "--config-yaml",
        required=True,
        help="GWDatasetGeneration yaml config used to generate --input "
        "(for the stored sample_rate, f_min highpass, and whiten.fftlength/average)",
    )
    parser.add_argument("--dest", required=True, help="output folder")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def plot_snr_scatter(stored_snr, recomputed_snr, dest_dir):
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(stored_snr, recomputed_snr, s=4, alpha=0.3, color="darkblue")
    lo, hi = 0, max(stored_snr.max(), recomputed_snr.max()) * 1.05
    axis.plot([lo, hi], [lo, hi], color="black", linestyle="dotted", linewidth=1)
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_xlabel("Stored SNR")
    axis.set_ylabel("Recomputed SNR (from whitened signal)")
    figure.tight_layout()
    output_path = os.path.join(dest_dir, "snr_scatter.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print("wrote", output_path, flush=True)


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.dest, exist_ok=True)

    config = load_config(arguments.config_yaml)
    general, whiten = config.general, config.whiten
    sample_rate = general.sample_rate / (general.downsample_rate or 1)
    highpass = general.f_min

    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=whiten.fftlength,
        average=whiten.average,
    )

    with h5py.File(arguments.input, "r") as h5_file:
        stored_snr = h5_file["snr"][:].astype(np.float64)
        num_events = h5_file["whitened_signal"].shape[0]
        n_samples = h5_file["whitened_signal"].shape[-1]
        n_psd_freq_bins = n_samples // 2 + 1

        recomputed_snr = np.empty(num_events)
        for start in range(0, num_events, arguments.batch_size):
            stop = min(start + arguments.batch_size, num_events)
            signal = torch.tensor(h5_file["whitened_signal"][start:stop], dtype=torch.float64)
            bkg = torch.tensor(h5_file["whitened_bkg"][start:stop], dtype=torch.float64)

            psd = spectral_density(bkg)
            psd = F.interpolate(psd, size=(n_psd_freq_bins,), mode="linear")

            with torch.no_grad():
                recomputed_snr[start:stop] = compute_network_snr(
                    signal, psd, sample_rate, highpass=highpass
                ).numpy()

    percent_diff = np.abs(recomputed_snr - stored_snr) / stored_snr * 100

    snrs_path = os.path.join(arguments.dest, "snrs.out")
    with open(snrs_path, "w") as snrs_file:
        snrs_file.write("index,stored_snr,recomputed_snr,percent_diff\n")
        for i, (stored, recomputed, diff) in enumerate(
            zip(stored_snr, recomputed_snr, percent_diff, strict=True)
        ):
            snrs_file.write(f"{i},{stored:.6f},{recomputed:.6f},{diff:.6f}\n")
    print("wrote", snrs_path, flush=True)

    summary_path = os.path.join(arguments.dest, "snr_summary.txt")
    with open(summary_path, "w") as summary_file:
        summary_file.write(f"input: {arguments.input}\n")
        summary_file.write(f"num_events: {num_events}\n")
        summary_file.write(
            f"max percent diff: {percent_diff.max():.6f}\n"
            f"mean percent diff: {percent_diff.mean():.6f}\n\n"
        )
        summary_file.write("threshold(%)  count  fraction\n")
        for threshold in THRESHOLDS:
            count = int((percent_diff <= threshold).sum())
            fraction = count / num_events
            summary_file.write(f"{threshold:<12g}  {count:<6d}  {fraction:.6f}\n")
    print("wrote", summary_path, flush=True)

    plot_snr_scatter(stored_snr, recomputed_snr, arguments.dest)


if __name__ == "__main__":
    main()
