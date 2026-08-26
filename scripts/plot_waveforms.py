"""Plot whitened strain from sig_combined_test.h5 in LIGO figure style.

Per event: one panel per interferometer covering the seconds around the
merger (8 s before by default). The
whitened data is drawn in grey underneath and the pure (noise-free) injected
signal on top in the observatory colour. The x axis is time relative to
coalescence, negative before the merger and 0 at it; the merger sits
`right_pad` seconds from the right edge of the window. Output is vector SVG.

Example:
    python plot_waveforms.py                       # events 0-2 + summary
    python plot_waveforms.py --events all
"""

import argparse
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwpy.plot.colors import GW_OBSERVATORY_COLORS
from gwpy.timeseries import TimeSeries

IFO_NAMES = ["H1", "L1"]
IFO_LABELS = ["Hanford", "Livingston"]
# official LIGO observatory colours (H1 red, L1 blue), used for the signal
IFO_COLORS = [GW_OBSERVATORY_COLORS[name] for name in IFO_NAMES]
DATA_COLOR = "0.8"          # light grey
DATA_ALPHA = 0.5

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(HERE, "sig_combined_test.h5"))
    parser.add_argument("--dest", default=os.path.join(HERE, "waveform_plots"))
    parser.add_argument("--events", default="0,1,2",
                        help='comma-separated indices, or "all"')
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    parser.add_argument("--right-pad", type=float, default=2.0,
                        help="coalescence position [s] from the right edge")
    parser.add_argument("--pre", type=float, default=8.0,
                        help="seconds before merger to show")
    parser.add_argument("--post", type=float, default=1.0,
                        help="seconds after merger to show")
    parser.add_argument("--bandpass", type=float, nargs=2, default=(30.0, 500.0),
                        help="bandpass applied for display only; 0 0 disables")
    parser.add_argument("--width", type=float, default=16.0, help="figure width [in]")
    parser.add_argument("--data-linewidth", type=float, default=0.25)
    parser.add_argument("--signal-linewidth", type=float, default=0.7)
    return parser.parse_args()


def bandpassed(x, fs, band):
    """Bandpass for display only; the stored data itself is already whitened."""
    if not band or band[0] <= 0:
        return x
    return TimeSeries(x.astype(np.float64), sample_rate=fs).bandpass(*band).value


def time_until_coalescence(n, fs, right_pad):
    """Time relative to the merger: negative before it, 0 at coalescence."""
    t_merge = n / fs - right_pad
    return np.arange(n) / fs - t_merge


def style_countdown_axis(ax, pre, post):
    """Ascending left to right, merger at 0."""
    ax.set_xlim(-pre, post)
    ax.grid(alpha=0.3)


def plot_event(h5, i, args, dest):
    fs = args.sample_rate
    n = h5["whitened_injected"].shape[-1]
    t = time_until_coalescence(n, fs, args.right_pad)
    sel = (t >= -args.pre) & (t <= args.post)

    injected = h5["whitened_injected"][i]
    signal = h5["whitened_signal"][i]
    snr = float(h5["snr"][i])
    mc = float(h5["chirp_mass"][i])

    figure, axes = plt.subplots(injected.shape[0], 1, sharex=True,
                                figsize=(args.width, 6.0))
    axes = np.atleast_1d(axes)
    for c, ax in enumerate(axes):
        inj = bandpassed(injected[c], fs, args.bandpass)
        sig = bandpassed(signal[c], fs, args.bandpass)
        ax.plot(t[sel], inj[sel], color=DATA_COLOR, lw=args.data_linewidth,
                alpha=DATA_ALPHA, label=f"{IFO_LABELS[c]} data (whitened)")
        ax.plot(t[sel], sig[sel], color=IFO_COLORS[c], lw=args.signal_linewidth,
                label="injected signal")
        ax.set_ylabel("Whitened strain")
        ax.legend(loc="upper left", fontsize=8)
        ax.text(0.995, 0.06, IFO_LABELS[c], ha="right", va="bottom",
                transform=ax.transAxes, fontsize=11, color=IFO_COLORS[c])
        style_countdown_axis(ax, args.pre, args.post)
    axes[-1].set_xlabel("Time until coalescence (s)")

    figure.suptitle(f"event {i}   network SNR = {snr:.2f}   "
                    rf"$\mathcal{{M}}_c$ = {mc:.3f} $M_\odot$")
    figure.tight_layout()
    path = os.path.join(dest, f"event_{i:03d}.svg")
    figure.savefig(path, format="svg")
    plt.close(figure)
    print("wrote", path, flush=True)


def plot_summary(h5, indices, args, dest):
    """All requested events' pure signals, stacked, Hanford only."""
    fs = args.sample_rate
    n = h5["whitened_injected"].shape[-1]
    t = time_until_coalescence(n, fs, args.right_pad)
    sel = (t >= -args.pre) & (t <= args.post)

    figure, axes = plt.subplots(len(indices), 1, sharex=True,
                                figsize=(args.width, 1.5 * len(indices) + 1))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, indices):
        sig = bandpassed(h5["whitened_signal"][i][0], fs, args.bandpass)
        ax.plot(t[sel], sig[sel], color=IFO_COLORS[0], lw=args.signal_linewidth)
        ax.set_ylabel(f"{i}", rotation=0, labelpad=14, va="center")
        ax.text(0.995, 0.85, f"SNR {float(h5['snr'][i]):.2f}", ha="right",
                va="top", transform=ax.transAxes, fontsize=8)
        style_countdown_axis(ax, args.pre, args.post)
    axes[-1].set_xlabel("Time until coalescence (s)")
    axes[0].set_title(f"{IFO_LABELS[0]} injected signals (whitened, bandpassed)")
    figure.tight_layout()
    path = os.path.join(dest, "summary_signals.svg")
    figure.savefig(path, format="svg")
    plt.close(figure)
    print("wrote", path, flush=True)


def main():
    args = parse_arguments()
    os.makedirs(args.dest, exist_ok=True)
    with h5py.File(args.input, "r") as h5:
        n_events = h5["whitened_injected"].shape[0]
        indices = (list(range(n_events)) if args.events == "all"
                   else [int(v) for v in args.events.split(",")])
        for i in indices:
            plot_event(h5, i, args, args.dest)
        plot_summary(h5, indices, args, args.dest)


if __name__ == "__main__":
    main()
