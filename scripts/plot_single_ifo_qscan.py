"""Single-interferometer figure: whitened strain above, Q-scan below.

The top panel shows the Hanford whitened data in grey with the injected signal
over it. The bottom panel is a Q-transform of the same data, with the
leading-order (0PN) chirp track drawn over it as a thick dotted white line, to
show where the signal sits in time-frequency even though it is invisible by eye.

To Newtonian/quadrupole order the gravitational-wave frequency of an inspiral
at a time tau before coalescence is

    f(tau) = (1 / pi) * (5 / (256 tau))**(3/8) * (G Mc / c**3)**(-5/8),

with Mc the (detector-frame) chirp mass. Output is vector SVG; only the
spectrogram mesh is rasterised, so text and curves stay vector.

Example:
    python plot_single_ifo_qscan.py
    python plot_single_ifo_qscan.py --event 7 --pre 6
"""

import argparse
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.size": 15.0,          # 50% up from the matplotlib default of 10
    "axes.labelsize": 15.0,
    "xtick.labelsize": 15.0,
    "ytick.labelsize": 15.0,
    "legend.fontsize": 15.0,
})
ANNOTATION_SIZE = 16.5          # 50% up from 11

# vivid amber: already maximal HSV saturation, hue nudged toward orange
# where the sRGB gamut allows more chroma, so it reads richer without darkening
SIGNAL_COLOR = "#ff8c00"
DATA_COLOR = "0.8"
DATA_ALPHA = 0.5

# G M_sun / c^3 in seconds
SOLAR_MASS_SECONDS = 4.925490947e-6

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(HERE, "sig_combined_test.h5"))
    parser.add_argument("--output", default=None,
                        help="defaults to single_ifo_qscan[_dark].svg next to this script")
    parser.add_argument("--dark", action="store_true",
                        help="transparent background with white axes, ticks, "
                             "labels, legend text and whitened data")
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--ifo", type=int, default=0, help="0 = Hanford")
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    parser.add_argument("--right-pad", type=float, default=2.0,
                        help="coalescence position [s] from the right edge")
    parser.add_argument("--pre", type=float, default=8.0)
    parser.add_argument("--post", type=float, default=1.0)
    parser.add_argument("--bandpass", type=float, nargs=2, default=(30.0, 500.0),
                        help="bandpass applied to the top panel only; 0 0 disables")
    parser.add_argument("--frange", type=float, nargs=2, default=(20.0, 1290.0),
                        help="upper bound is where gwpy's Q-transform tiles stop")
    parser.add_argument("--width", type=float, default=16.0)
    parser.add_argument("--data-linewidth", type=float, default=0.25)
    parser.add_argument("--signal-linewidth", type=float, default=0.7)
    return parser.parse_args()


def apply_dark_style(figure, axes, legend):
    """Invert the black furniture to white for use over a dark background."""
    for panel in axes:
        panel.tick_params(colors="white", which="both")
        panel.xaxis.label.set_color("white")
        panel.yaxis.label.set_color("white")
        for spine in panel.spines.values():
            spine.set_color("white")
    for entry in legend.get_texts():
        entry.set_color("white")
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_edgecolor("white")
    figure.patch.set_alpha(0.0)


def bandpassed(x, fs, band):
    """Bandpass for display only; the stored data itself is already whitened."""
    if not band or band[0] <= 0:
        return x
    return TimeSeries(x.astype(np.float64), sample_rate=fs).bandpass(*band).value


def chirp_track(chirp_mass, tau):
    """0PN gravitational-wave frequency at a time `tau` before coalescence."""
    mc_seconds = chirp_mass * SOLAR_MASS_SECONDS
    return (5.0 / (256.0 * tau)) ** (3.0 / 8.0) * mc_seconds ** (-5.0 / 8.0) / np.pi


def main():
    args = parse_arguments()
    if args.output is None:
        name = "single_ifo_qscan_dark.svg" if args.dark else "single_ifo_qscan.svg"
        args.output = os.path.join(HERE, name)
    fs = args.sample_rate
    data_color = "white" if args.dark else DATA_COLOR

    with h5py.File(args.input, "r") as h5:
        injected = h5["whitened_injected"][args.event, args.ifo].astype(np.float64)
        signal = h5["whitened_signal"][args.event, args.ifo].astype(np.float64)
        chirp_mass = float(h5["chirp_mass"][args.event])

    n = len(injected)
    t_merge = n / fs - args.right_pad
    t = np.arange(n) / fs - t_merge
    sel = (t >= -args.pre) & (t <= args.post)

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(args.width, 7.0))
    caxes = []
    for panel in axes:
        cax = make_axes_locatable(panel).append_axes('right', size='1.6%', pad=0.12)
        caxes.append(cax)
    for cax in caxes:            # kept only so both panels stay the same width
        cax.axis('off')

    # --- top: whitened strain, no legend -------------------------------------
    ax = axes[0]
    ax.plot(t[sel], bandpassed(injected, fs, args.bandpass)[sel],
            color=data_color, lw=args.data_linewidth, alpha=DATA_ALPHA,
            label="Whitened Data")
    ax.plot(t[sel], bandpassed(signal, fs, args.bandpass)[sel],
            color=SIGNAL_COLOR, lw=args.signal_linewidth, label="BNS Merger Signal")
    ax.set_ylabel("Strain")
    legend = ax.legend(loc="upper left")
    ax.text(0.995, 0.06, "Single IFO Response", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=ANNOTATION_SIZE, fontweight="bold",
            color=SIGNAL_COLOR)
    ax.grid(alpha=0.3)

    # --- bottom: Q-scan with the analytic chirp track ------------------------
    ax = axes[1]
    ts = TimeSeries(injected, sample_rate=fs, t0=-t_merge)
    q = ts.q_transform(outseg=(-args.pre, args.post), whiten=False,
                       frange=tuple(args.frange), logf=True)
    # gwpy caps the Q-transform at nyquist / (1 + sqrt(11) / Q), so the tiles
    # stop short of Nyquist; paint the gap with the colormap's zero colour
    # rather than leaving bare canvas.
    cmap = plt.get_cmap("viridis")
    ax.set_facecolor(cmap(0.0))
    mesh = ax.pcolormesh(q.times.value, q.frequencies.value, q.value.T,
                         cmap=cmap, shading="auto", rasterized=True)

    tau = np.logspace(np.log10(1.0 / fs), np.log10(args.pre), 512)
    freq = chirp_track(chirp_mass, tau)
    inside = (freq >= args.frange[0]) & (freq <= args.frange[1])
    ax.plot(-tau[inside], freq[inside], color="white", lw=3.0, ls=":",
            solid_capstyle="round")
    tau_label = 0.62 * args.pre
    ax.text(-tau_label, chirp_track(chirp_mass, tau_label) * 1.75,
            "Expected Signal (invisible by eye)", color="white",
            fontsize=ANNOTATION_SIZE, fontweight="bold", ha="left", va="bottom")

    # inset colourbar: 2/5 panel height, sitting in the 0-1 s corner
    x_zero = (0.0 + args.pre) / (args.pre + args.post)
    bar_x, bar_y, bar_w, bar_h = x_zero + 0.024, 0.11, 0.010, 0.40
    cbar_axis = ax.inset_axes([bar_x, bar_y, bar_w, bar_h])
    colorbar = figure.colorbar(mesh, cax=cbar_axis)
    colorbar.set_ticks([t for t in (0, 5, 10, 15, 20) if t <= mesh.get_clim()[1]])
    colorbar.outline.set_edgecolor("white")
    cbar_axis.tick_params(colors="white", labelsize=11)
    ax.text(bar_x + bar_w + 0.034, bar_y + bar_h / 2, "normalized energy",
            color="white", rotation=90, ha="center", va="center", fontsize=11,
            transform=ax.transAxes)

    ax.set_yscale("log")
    ax.set_ylim(*args.frange)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time until Coalescence (s)")
    ax.set_xlim(-args.pre, args.post)
    axes[0].set_xlim(-args.pre, args.post)

    figure.align_ylabels(axes)
    if args.dark:
        apply_dark_style(figure, axes, legend)
    figure.tight_layout()
    figure.savefig(args.output, format="svg", transparent=args.dark)
    plt.close(figure)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
