from ml4gw.transforms import SpectralDensity
import torch
from pathlib import Path
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import Whiten
from utils import load_config
from waveforms import generate_signals
from ml4gw.gw import compute_network_snr,reweight_snrs
import importlib

def injection(config, data_dir: str, device: str, inject: bool):

    ifos = config.general.ifos                        # number of observatories
    batch_size = config.general.batch_size            # batch size
    sample_rate = config.general.sample_rate          # sample rate of time-domain data (Hz)
    bkg_sample_rate = config.general.bkg_sample_rate  # sample rate of background data (Hz)
    f_min = config.general.f_min                      # minimum frequency (for highpass)
    kernel_length = config.general.waveform_duration  # waveform duration (sec)

    # Length of filter. A segment of length fduration / 2
    # will be cropped from either side after whitening
    fduration = config.whiten.fduration               # (gwpy doc) Duration (in seconds) of the time-domain FIR whitening filter, must be no longer than fftlength, default: 2 seconds.
    fftlength = config.whiten.fftlength               # freq resolution = 1 / fftlength
    psd_length = config.whiten.psd_length             # length of PSD sample (sec)
    overlap = config.whiten.overlap                   # (gwpy doc) Overlap between windows used for FFT calculation. If left as ``None``, this will be set to ``fftlength / 2``.
    average = config.whiten.average                   # (gwpy doc) Aggregation method to use for combining windowed FFTs. Allowed values are ``"mean"`` and ``"median"``.

    psd_size = int(psd_length * sample_rate)          # 
    kernel_size = int(kernel_length * sample_rate)

    # Total length of data to sample
    window_length = psd_length + fduration + kernel_length # (sec); adding fduration to add pad of length fduration/2 at each side
    num_samples = int(config.general.waveform_duration * sample_rate)
    num_freqs = num_samples // 2 + 1

    fnames = list(data_dir.iterdir())    # background samples
    dataloader = Hdf5TimeSeriesDataset(
        fnames=fnames,
        channels=ifos,
        kernel_size=int(window_length * bkg_sample_rate),
        batch_size=batch_size,  
        batches_per_epoch=1,  # Just doing 1 here for demonstration purposes
        coincident=False,     # random shift
    )

    background_samples = next(iter(dataloader)).to(device)
    background_samples = background_samples[::]
    #print(background_samples.shape)

    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        overlap=overlap,
        average=average,
    ).to(device)

    whiten = Whiten(
        fduration=fduration, sample_rate=sample_rate, highpass=f_min
    ).to(device)

    psd = spectral_density(background_samples[..., :psd_size].double())
    #print(f"PSD shape: {psd.shape}")
    kernel = background_samples[..., psd_size:]

    if inject:
        waveforms, params = generate_signals(config, device, save=False) 

        pad = int(fduration / 2 * sample_rate)
        injected = kernel.detach().clone()

        # calculation and reweighting of SNRs
        if psd.shape[-1] != num_freqs:
            # interpolate requires at least 3D input [B, C, L]
            # prepend dummy dims to satisfy this
            # since only the last dimension (L) is resized
            while psd.ndim < 3:
                psd = psd[None]
            psd = torch.nn.functional.interpolate(psd, size=(num_freqs,), mode='linear')

        func_path = config.snr_reweighting.func
        module_name, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        args = config.snr_reweighting.args
        target_snrs = func(*args).sample((batch_size,)).to(device)

        waveforms = reweight_snrs(responses=waveforms, target_snrs=target_snrs, psd=psd, sample_rate=sample_rate, highpass=f_min,)

        raw_bkg = injected.clone()
        injected[:, :, pad:-pad] += waveforms[..., -kernel_size:]

        ### WHITENING
        whitened_injected = whiten(injected, psd)

        # compute network SNR
        network_snr = compute_network_snr(responses=waveforms, psd=psd, sample_rate=sample_rate, highpass=f_min)
        params['snr'] = network_snr

        # Compute whitened signal
        raw_signal = torch.zeros_like(kernel)
        raw_signal[:, :, pad:-pad] += waveforms[..., -kernel_size:]
        whitened_signal = whiten(raw_signal, psd)

        # Compute whitened background
        whitened_bkg = whiten(raw_bkg, psd)
    else:
        whitened_injected = whiten(kernel, psd)
        params = None
        whitened_signal = None

    return whitened_injected, whitened_signal, raw_signal, whitened_bkg, raw_bkg, params

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path='config.yaml')
    data_dir = Path("./data")
    # And this to the directory where you want to download the data
    background_dir = data_dir / "background_data"

    injection(config, data_dir=background_dir, device=device, inject=True)
