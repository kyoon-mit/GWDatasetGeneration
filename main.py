from utils import load_config
from injections import injection
import torch
import numpy as np
import argparse
from pathlib import Path
import h5py
import os
import gc
from tqdm import tqdm
from gwpy.timeseries import TimeSeries

def _is_valid(whitened_signal, raw_signal):
    for t in (whitened_signal, raw_signal):
        if torch.isnan(t).any():
            return False
        if (t.abs().sum(dim=(-2, -1)) == 0).any():
            return False
    return True

def _resample(tensor, orig_freq, new_freq):
    arr = tensor.cpu().numpy()
    return np.array([[TimeSeries(arr[b, c], sample_rate=orig_freq).resample(new_freq).value
                      for c in range(arr.shape[1])] for b in range(arr.shape[0])])

def main(config_path: str, data_dir: str, output_dir: str, num_waveforms: int = None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    if num_waveforms is not None:
        config.general.num_waveforms = num_waveforms

    if config.general.downsample_rate is None:
        downsample_rate = 1
    else:
        downsample_rate = config.general.downsample_rate

    data_dir = Path(data_dir)
    out_dir = Path(output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    total = 0
    consecutive_invalid = 0
    max_consecutive_invalid = 10
    with tqdm(total=config.general.num_waveforms, desc="Processing", unit="step") as pbar:
        while total<config.general.num_waveforms:
            # generate signals
            outfile = out_dir / 'sig_{0}.h5'.format(total)      # sig + bkg

            if not outfile.exists():
                whitened_injected, whitened_signal, raw_signal, whitened_bkg, raw_bkg, params =\
                    injection(config, data_dir=data_dir, device=device, inject=True)

                if not _is_valid(whitened_signal, raw_signal):
                    consecutive_invalid += 1
                    del whitened_injected, whitened_signal, whitened_bkg, raw_signal, raw_bkg, params
                    gc.collect()
                    torch.cuda.empty_cache()
                    if consecutive_invalid >= max_consecutive_invalid:
                        raise RuntimeError(
                            f"Aborting: {consecutive_invalid} consecutive batches contained "
                            "only NaN or zero signals. Check waveform generation parameters."
                        )
                    continue

                consecutive_invalid = 0

                orig_freq = config.general.sample_rate
                new_freq = orig_freq // downsample_rate
                # Shape: (B, nifos, L)
                with h5py.File(outfile, 'w') as h5f:
                    h5f.create_dataset('whitened_injected', data=_resample(whitened_injected, orig_freq, new_freq))
                    h5f.create_dataset('whitened_signal', data=_resample(whitened_signal, orig_freq, new_freq))
                    h5f.create_dataset('whitened_bkg', data=_resample(whitened_bkg, orig_freq, new_freq))
                    h5f.create_dataset('raw_signal', data=_resample(raw_signal, orig_freq, new_freq))
                    h5f.create_dataset('raw_bkg', data=_resample(raw_bkg, orig_freq, new_freq))
                    # h5f.create_dataset('whitened_injected', data=whitened_injected.cpu().numpy())
                    # h5f.create_dataset('whitened_signal', data=whitened_signal.cpu().numpy())
                    # h5f.create_dataset('whitened_bkg', data=whitened_bkg.cpu().numpy())
                    # h5f.create_dataset('raw_signal', data=raw_signal.cpu().numpy())
                    # h5f.create_dataset('raw_bkg', data=raw_bkg.cpu().numpy())
                    for k in params.keys():
                        h5f.create_dataset(k, data=params[k].cpu().numpy())

                del whitened_injected, whitened_signal, whitened_bkg, raw_signal, raw_bkg, params
                gc.collect()
                torch.cuda.empty_cache()

            # generate backgrounds
            # backgrounds, _ = injection(config, data_dir=data_dir, device=device, inject=False)
            # bkg_data = backgrounds.cpu().numpy()
            # with h5py.File(out_dir / 'bkg_{0}.h5'.format(total), 'w') as h5f:
            #     h5f.create_dataset('data', data=bkg_data)
            # del backgrounds, bkg_data
            # gc.collect()
            # torch.cuda.empty_cache()

            total += config.general.batch_size
            pbar.update(config.general.batch_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Data generation script for GW signals"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to folder containing public data"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to output folder for .hdf5 files"
    )
    parser.add_argument(
        "--num-waveforms",
        type=int,
        default=None,
        help="Number of waveforms to generate (overrides config)"
    )
    args = parser.parse_args()

    main(config_path=args.config, data_dir=args.data, output_dir=args.out, num_waveforms=args.num_waveforms)
