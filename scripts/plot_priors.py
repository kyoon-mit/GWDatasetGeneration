"""Plot the raw parameter distributions from a GWDatasetGeneration combined
sig_combined_*.h5 file (flat top-level datasets: mass_1, chirp_mass, snr, ...
plus whitened_signal/whitened_bkg/whitened_injected — no "parameters" group,
no h5 attrs). Adapted from aframe/dev/data/diagnostic/plot_priors_train.py.

Makes one histogram per parameter (raw counts, not normalized) using the bin
range, bin size, color, and alpha from ../configs/plot_configs.json, and
writes a metadata.info file describing the input (read from the sibling
GWDatasetGeneration yaml config, since the h5 itself carries no metadata).

Example:
    python plot_priors.py \\
        --input /path/to/sig_combined_train.h5 \\
        --config-yaml /path/to/BNS_SNR7-12_TRAIN_VAL_2048Hz_BNS.yaml \\
        --dest  /path/to/train/prior_plots
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_config

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "plot_configs.json"
)

# datasets in the combined h5 that are not sampled parameters
NON_PARAMETER_KEYS = {"whitened_signal", "whitened_bkg", "whitened_injected"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="sig_combined_*.h5 file")
    parser.add_argument("--dest", required=True, help="output plot folder")
    parser.add_argument(
        "--config-yaml",
        default=None,
        help="GWDatasetGeneration yaml config used to generate --input "
        "(for metadata.info); defaults to the single *.yaml sitting "
        "next to --input's dataset directory, if any.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="json with per-parameter bins/color/alpha",
    )
    return parser.parse_args()


def plot_parameters(h5_file, parameter_config, dest_dir):
    """One raw-count histogram per parameter, saved as <name>.png."""
    parameter_names = [k for k in h5_file.keys() if k not in NON_PARAMETER_KEYS]

    for parameter_name in parameter_names:
        values = h5_file[parameter_name][:]
        settings = parameter_config.get(parameter_name)
        if settings is None:
            minimum, maximum = float(values.min()), float(values.max())
            settings = {
                "min": minimum,
                "max": maximum,
                "bin_size": (maximum - minimum) / 50 or 1e-2,
                "color": "blue",
                "alpha": 1.0,
                "xlabel": parameter_name,
                "label": parameter_name,
            }
        bin_edges = np.arange(
            settings["min"],
            settings["max"] + settings["bin_size"],
            settings["bin_size"],
        )

        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(values, bins=bin_edges, color=settings["color"], alpha=settings["alpha"])
        axis.set_xlabel(settings["xlabel"])
        axis.set_ylabel("count")
        figure.tight_layout()
        output_path = os.path.join(dest_dir, f"{parameter_name}.png")
        figure.savefig(output_path, dpi=140)
        plt.close(figure)
        print("wrote", output_path, flush=True)


def find_sibling_config(input_path):
    """Look for a single *.yaml next to the train/val/test dir containing input_path."""
    dataset_dir = Path(input_path).resolve().parent.parent
    yamls = list(dataset_dir.glob("*.yaml"))
    if len(yamls) == 1:
        return str(yamls[0])
    return None


def write_metadata(h5_file, input_path, config_yaml_path, dest_dir):
    """Write shape/config info about the input to metadata.info."""
    parameter_names = [k for k in h5_file.keys() if k not in NON_PARAMETER_KEYS]
    length = h5_file[parameter_names[0]].shape[0]

    lines = [
        f"file_name: {os.path.basename(input_path)}",
        f"num_injections: {length}",
        f"h5_keys: {list(h5_file.keys())}",
    ]
    if "whitened_signal" in h5_file:
        batch, num_ifos, num_samples = h5_file["whitened_signal"].shape
        lines.append(f"whitened_signal_shape: {(batch, num_ifos, num_samples)}")

    if config_yaml_path is not None:
        config = load_config(config_yaml_path)
        general = config.general
        sample_rate = general.sample_rate / (general.downsample_rate or 1)
        lines += [
            f"config_yaml: {config_yaml_path}",
            f"waveform_duration: {general.waveform_duration}",
            f"right_pad: {general.right_pad}",
            f"generation_sample_rate: {general.sample_rate}",
            f"downsample_rate: {general.downsample_rate}",
            f"stored_sample_rate: {sample_rate}",
            f"f_min: {general.f_min}",
            f"snr_reweighting: {config.snr_reweighting.func} {config.snr_reweighting.args}",
        ]
    else:
        lines.append("config_yaml: not found")

    metadata_path = os.path.join(dest_dir, "metadata.info")
    with open(metadata_path, "w") as metadata_file:
        metadata_file.write("\n".join(lines) + "\n")
    print("wrote", metadata_path, flush=True)


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.dest, exist_ok=True)
    with open(arguments.config) as config_file:
        parameter_config = json.load(config_file)["parameters"]

    config_yaml_path = arguments.config_yaml or find_sibling_config(arguments.input)

    with h5py.File(arguments.input, "r") as h5_file:
        plot_parameters(h5_file, parameter_config, arguments.dest)
        write_metadata(h5_file, arguments.input, config_yaml_path, arguments.dest)


if __name__ == "__main__":
    main()
