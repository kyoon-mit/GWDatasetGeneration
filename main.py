from utils import load_config
from injections import injection
import torch
import argparse
from pathlib import Path
import h5py
import os
import gc
from tqdm import tqdm

def main(config_path: str, data_dir: str, output_dir: str, num_waveforms: int = None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    if num_waveforms is not None:
        config.general.num_waveforms = num_waveforms

    data_dir = Path(data_dir)
    out_dir = Path(output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    total = 0
    with tqdm(total=config.general.num_waveforms, desc="Processing", unit="step") as pbar:
        while total<config.general.num_waveforms:
            # generate signals
            outfile = out_dir / 'sig_{0}.h5'.format(total)      # sig + bkg

            if not outfile.exists():
                whitened_injected, whitened_signal, params = injection(config, data_dir=data_dir, device=device, inject=True)
                # whitened_bkg, _, _ = injection(config, data_dir=data_dir, device=device, inject=False)
                injected_data = whitened_injected.cpu().numpy()
                sig_data = whitened_signal.cpu().numpy()
                # bkg_data = whitened_bkg.cpu().numpy()
                bkg_data = injected_data - sig_data
                with h5py.File(outfile, 'w') as h5f:
                    h5f.create_dataset('injected_data', data=injected_data)
                    h5f.create_dataset('sig_only_data', data=sig_data)
                    h5f.create_dataset('bkg_only_data', data=bkg_data)
                    for k in params.keys():
                        h5f.create_dataset(k, data=params[k].cpu().numpy())

                del whitened_injected, injected_data, params
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
