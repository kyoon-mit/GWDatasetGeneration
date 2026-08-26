#!/usr/bin/env python3
"""
SNR Segment Check
=================
Compute ml4gw network SNR over user-defined time windows of whitened GW signals
and compare against the SNR stored in the HDF5 file.

Usage
-----
    python check_snr_segments.py --h5 <path/to/sig_combined.h5> [options]

Notes
-----
The stored SNR was computed before downsampling at the original sample rate
(e.g. 4096 Hz, covering 20–2048 Hz).  If the HDF5 file stores 256 Hz data,
computed SNR covers only 20–128 Hz, so computed/stored ≈ 0.60–0.65 is expected.
At 4096 Hz, computed/stored ≈ 1.0 is expected for the full-segment window.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ml4gw.gw import compute_network_snr
from ml4gw.transforms import SpectralDensity
from tqdm import tqdm


# ── default time windows ──────────────────────────────────────────────────────
DEFAULT_WINDOWS = [
    (0,  64, "0–64 s  (full segment)"),
    (59, 64, "59–64 s (5 s incl. merger)"),
    (0,  55, "0–55 s  (pre-merger)"),
    (63, 64, "63–64 s (1 s merger)"),
]


def compute_segment_snr(
    h5_path: str,
    spectral_density: SpectralDensity,
    start_s: float,
    end_s: float,
    sample_rate: int,
    f_min: float,
    n_events: int | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute network SNR for the whitened signal over [start_s, end_s].

    For each batch, the PSD is estimated from the corresponding whitened_bkg
    entries, matching how injections.py computes the stored SNR.

    Returns
    -------
    snr_computed : (n_events,) ndarray
    snr_stored   : (n_events,) ndarray
    """
    start_sample    = int(start_s * sample_rate)
    end_sample      = int(end_s   * sample_rate)
    n_psd_freq_bins = (end_sample - start_sample) // 2 + 1

    with h5py.File(h5_path, "r") as f:
        n_total  = f["snr"].shape[0]
        n_events = min(n_events or n_total, n_total)
        snr_stored = f["snr"][:n_events].astype(np.float64)

        snr_chunks = []
        for start in tqdm(range(0, n_events, batch_size),
                          desc=f"  SNR [{start_s:.0f}–{end_s:.0f}s]", leave=False):
            end    = min(start + batch_size, n_events)
            bkg    = torch.tensor(f["whitened_bkg"][start:end], dtype=torch.float64)
            window = torch.tensor(
                f["whitened_signal"][start:end, :, start_sample:end_sample],
                dtype=torch.float64,
            )                                       # (batch, n_ifos, n_signal_samples)

            # Per-event PSD from whitened_bkg, interpolated to window's FFT grid.
            # Stay in float64: float32 underflows on PSD values ~1e-49 → NaN.
            psd = spectral_density(bkg)             # (batch, n_ifos, n_psd_bins)
            psd_window = F.interpolate(
                psd, size=(n_psd_freq_bins,), mode="linear"
            )                                       # (batch, n_ifos, n_psd_freq_bins)

            with torch.no_grad():
                snr_chunks.append(
                    compute_network_snr(
                        responses=window,
                        psd=psd_window,
                        sample_rate=float(sample_rate),
                        highpass=float(f_min),
                    ).numpy()
                )

    snr_computed = np.concatenate(snr_chunks)
    return snr_computed, snr_stored


def plot_scatter(results: dict, windows: list, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 4.5))
    if len(windows) == 1:
        axes = [axes]

    for ax, (_, _, label) in zip(axes, windows):
        snr_c, snr_s = results[label]
        valid = np.isfinite(snr_c) & np.isfinite(snr_s)
        snr_c, snr_s = snr_c[valid], snr_s[valid]

        lo = min(snr_c.min(), snr_s.min()) * 0.9
        hi = max(snr_c.max(), snr_s.max()) * 1.1

        ax.scatter(snr_s, snr_c, s=4, alpha=0.3, color="steelblue", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Stored SNR", fontsize=9)
        ax.set_ylabel("Computed SNR", fontsize=9)
        ax.set_title(f"{label}\n({valid.sum()} / {len(valid)} valid)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, lw=0.3)

    plt.suptitle("Computed vs stored network SNR per time window", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved scatter plot → {out_path}")


def plot_hist(results: dict, windows: list, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 4))
    if len(windows) == 1:
        axes = [axes]

    for ax, (_, _, label) in zip(axes, windows):
        snr_c, snr_s = results[label]
        ratio = snr_c / snr_s
        ax.hist(ratio, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(1.0, color="red", lw=1.2, ls="--", label="ratio = 1")
        ax.axvline(np.nanmedian(ratio), color="orange", lw=1.2,
                   label=f"median={np.nanmedian(ratio):.3f}")
        ax.set_xlabel("SNR_computed / SNR_stored", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, lw=0.3)

    plt.suptitle("SNR ratio (computed / stored) per time window", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved histogram → {out_path}")


def plot_fraction(results: dict, windows: list, out_path: Path) -> None:
    label_full = windows[0][2]
    snr_full   = results[label_full][0]
    sub_windows = windows[1:]

    fig, axes = plt.subplots(1, len(sub_windows), figsize=(5 * len(sub_windows), 4))
    if len(sub_windows) == 1:
        axes = [axes]

    for ax, (_, _, label) in zip(axes, sub_windows):
        frac = (results[label][0] / snr_full) ** 2
        ax.hist(frac, bins=60, color="darkorange", edgecolor="none", alpha=0.85)
        ax.axvline(np.nanmedian(frac), color="red", lw=1.2, ls="--",
                   label=f"median SNR² frac={np.nanmedian(frac):.3f}")
        ax.set_xlabel("SNR²_window / SNR²_full", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, lw=0.3)

    plt.suptitle("Fraction of total SNR² in each sub-window", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved fraction plot → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5",          required=True,  help="Path to HDF5 file")
    parser.add_argument("--out_dir",     default=".",    help="Directory for output plots (default: .)")
    parser.add_argument("--sample_rate", type=int,   default=256,  help="Sample rate of stored data (default: 256)")
    parser.add_argument("--f_min",       type=float, default=20.0, help="Highpass frequency in Hz (default: 20)")
    parser.add_argument("--fftlength",   type=float, default=2.0,  help="SpectralDensity fftlength in s (default: 2)")
    parser.add_argument("--n_events",    type=int,   default=500,  help="Events to process (default: 500)")
    parser.add_argument("--batch_size",  type=int,   default=32,   help="Events per batch — reduce if OOM (default: 32)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spectral_density = SpectralDensity(
        sample_rate=args.sample_rate,
        fftlength=args.fftlength,
        average="median",
    )
    n_freq_bins = int(args.fftlength * args.sample_rate) // 2 + 1
    print(f"SpectralDensity: fftlength={args.fftlength}s  →  {n_freq_bins} freq bins")

    with h5py.File(args.h5, "r") as f:
        n_total  = f["snr"].shape[0]
        n_events = min(args.n_events, n_total)
    print(f"Events: {n_events} / {n_total}")

    # ── compute SNR per window ────────────────────────────────────────────────
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for start_s, end_s, label in DEFAULT_WINDOWS:
        print(f"\nWindow: {label}")
        snr_c, snr_s = compute_segment_snr(
            h5_path=args.h5,
            spectral_density=spectral_density,
            start_s=start_s,
            end_s=end_s,
            sample_rate=args.sample_rate,
            f_min=args.f_min,
            n_events=n_events,
            batch_size=args.batch_size,
        )
        results[label] = (snr_c, snr_s)
        ratio = np.nanmedian(snr_c / snr_s)
        print(f"  computed — mean={snr_c.mean():.2f}  std={snr_c.std():.2f}  median={np.median(snr_c):.2f}")
        print(f"  stored   — mean={snr_s.mean():.2f}  std={snr_s.std():.2f}  median={np.median(snr_s):.2f}")
        print(f"  ratio (computed/stored) median={ratio:.3f}")

    plot_scatter(  results, DEFAULT_WINDOWS, out_dir / "snr_scatter.png")
    plot_hist(     results, DEFAULT_WINDOWS, out_dir / "snr_ratio_hist.png")
    plot_fraction( results, DEFAULT_WINDOWS, out_dir / "snr_fraction.png")


if __name__ == "__main__":
    main()
