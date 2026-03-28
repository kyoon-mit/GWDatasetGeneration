# Dataset generation for gravitational-wave physics

This repo is meant to generate data for ML tasks in gravitational wave physics. It is based on `ml4gw` and primarily uses its functionality.

To generate data, run
```python
python main.py --config config.yaml --data path/to/data/folder --out path/to/output/directory
```
with
- `config`: This is the config file containing all general and signal-specific setups. In the `config` folder, you can find example configs for BBH and BNS signals.
- `data`: path to directory containing open (background) data. The data can be downloaded using the `load_data.py` script.
- `out`: output directory where to store the dataset.

To update the number of generated signal and background events, change the `num_waveforms` parameter in the `config.yaml` file.

The generated output are `sig.h5` and `bkg.h5` files, which contain the time series for the two detectors (`data`) to be used for classification. The `sig.h5` files additionally contain all parameters defining the signal waveform to be used in regression tasks or for more detailed studies.

---

## Configuration reference

All generation behaviour is controlled by a single YAML config file. Example configs are in `configs/`.

### `general`

| Key | Description |
|---|---|
| `type` | Waveform approximant family. One of `BNS`, `BNS_IMRPhenomPv2`, or `BBH` (see Waveform approximants below). |
| `num_waveforms` | Total number of events to generate. Can be overridden at runtime with `--num-waveforms`. |
| `waveform_duration` | Duration of the output waveform window in seconds (e.g. 64). This is the length of the **whitened** output; the raw arrays are `waveform_duration + fduration` seconds. |
| `sample_rate` | Native sample rate in Hz used for waveform generation and whitening (e.g. 4096). |
| `bkg_sample_rate` | Sample rate of the background HDF5 files on disk. Must match the files downloaded by `load_data.py`. |
| `downsample_rate` | Integer stride applied to the time axis when saving to HDF5: `[..., ::downsample_rate]`. Effective output rate = `sample_rate / downsample_rate`. Set to 1 to disable. |
| `right_pad` | Seconds from the right edge of the window to the coalescence point. Controls where in the window the merger sits. |
| `f_min` | Lower frequency cutoff in Hz for whitening highpass and SNR calculation. |
| `f_max` | Upper frequency cutoff in Hz (used for waveform generation). |
| `f_ref` | Reference frequency in Hz for waveform phasing. |
| `ifos` | List of interferometers, e.g. `['H1', 'L1']`. |
| `batch_size` | Number of events generated per iteration. Each batch is saved as one `sig_N.h5` file. |

### `waveform`

Each entry defines a prior distribution for one waveform parameter. The format is:

```yaml
param_name:
    func: module.ClassName   # any importable distribution with a .sample() method
    args: [arg1, arg2, ...]  # positional args to the constructor
                             # string args are resolved as references to already-sampled params
```

**Intrinsic parameters** (passed to the waveform generator):

| Key | Description |
|---|---|
| `mass_1`, `mass_2` | Component masses in solar masses. For `BNS`/`BNS_IMRPhenomPv2`, `waveforms.py` sorts these so that `mass_1 >= mass_2` — both can be drawn from the same distribution. |
| `s1z`, `s2z` | Aligned spin components along the orbital angular momentum axis (used by `BNS` / TaylorF2). |
| `a_1`, `a_2` | Spin magnitudes in [0, 0.998] (used by `BNS_IMRPhenomPv2`). |
| `tilt_1`, `tilt_2` | Spin tilt angles (polar, in radians). `ml4gw.distributions.Sine` gives an isotropic prior. |
| `phi_12` | Azimuthal angle between the two spin vectors (radians). |
| `phi_jl` | Azimuthal angle of the orbital angular momentum relative to the total angular momentum (radians). |
| `distance` | Luminosity distance in Mpc. `ml4gw.distributions.PowerLaw` with exponent 2 gives a uniform-in-volume prior. |
| `phic` | Orbital phase at coalescence (radians). Use `Uniform(0, 2π)` for a physically representative dataset. |
| `inclination` | Angle between the orbital angular momentum and the line of sight. `ml4gw.distributions.Sine` gives isotropic. |

**Extrinsic / projection parameters** (optional — if omitted, isotropic defaults are used):

| Key | Default | Description |
|---|---|---|
| `dec` | `Cosine()` | Declination (radians). Cosine prior = isotropic sky. |
| `psi` | `Uniform(0, π)` | Polarisation angle (radians). |
| `phi` | `Uniform(-π, π)` | Right ascension proxy (radians). |

If any of `dec`, `psi`, `phi` are defined in the `waveform` block they override the defaults.

#### Mass prior strategies

- **Wide / training prior** (`mpriorsetting1`): both masses drawn from `Uniform(0.8, 2.8)` independently; `waveforms.py` sorts so `m1 ≥ m2`.
- **Realistic / test prior** (`mprior_realistic`): `mass_1 ~ Triangular(min=1.0, max=2.5, mode=2.5)`, `mass_2 ~ Uniform(1.0, mass_1)` (conditional). This matches the bilby convention used in GW parameter estimation. Use this prior for test sets only; use the wide prior for train/val so the model is not penalised for in-distribution generalisation.

### `whiten`

| Key | Description |
|---|---|
| `fftlength` | FFT segment length in seconds. Frequency resolution = `1 / fftlength`. |
| `overlap` | Overlap between FFT windows. `null` defaults to `fftlength / 2`. |
| `average` | Aggregation method for Welch PSD: `median` (recommended) or `mean`. |
| `psd_length` | Seconds of background data used to estimate the PSD before each batch. |
| `fduration` | Duration of the time-domain FIR whitening filter in seconds. The whitener crops `fduration / 2` from each end, so raw arrays are `waveform_duration + fduration` seconds while whitened arrays are exactly `waveform_duration` seconds. |

### `snr_reweighting`

Defines the target SNR distribution. Waveforms are rescaled after injection so the network SNR matches a sample from this distribution.

```yaml
snr_reweighting:
    func: ml4gw.distributions.DeltaFunction
    args: [50]           # fixed SNR=50

snr_reweighting:
    func: ml4gw.distributions.PowerLaw
    args: [5, 50]        # SNR ~ PowerLaw between 5 and 50
```

---

## Waveform approximants

Controlled by `general.type`:

| `type` | Approximant | Spin model | Notes |
|---|---|---|---|
| `BNS` | TaylorF2 | Aligned spin (`s1z`, `s2z`) | Frequency-domain PN; fast, accurate for BNS. Requires `chi1`/`chi2` aliases. |
| `BNS_IMRPhenomPv2` | IMRPhenomPv2 | Precessing spin (`s1x/y/z`, `s2x/y/z`) | Full precession. Spin Cartesian components are computed from `(a, tilt, phi_12, phi_jl)` in `waveforms.py`. |
| `BBH` (default) | IMRPhenomPv2 | Precessing spin | Takes `chirp_mass`, `mass_ratio`, `chi1/chi2` directly. |

---

## Output HDF5 datasets

Each `sig_N.h5` file contains one batch of `batch_size` events with the following datasets, all of shape `(batch_size, n_ifos, L)` unless noted:

| Dataset | Shape | Description |
|---|---|---|
| `whitened_injected` | `(B, 2, L_wht)` | Whitened strain with signal injected into real background noise. Primary input for ML models. |
| `whitened_signal` | `(B, 2, L_wht)` | Whitened signal-only (no noise). Useful for SNR studies. |
| `whitened_bkg` | `(B, 2, L_wht)` | Whitened background-only (no signal). |
| `raw_signal` | `(B, 2, L_raw)` | Unwhitened signal in detector frame. `L_raw = (waveform_duration + fduration) * effective_rate`. |
| `raw_bkg` | `(B, 2, L_raw)` | Unwhitened background segment. |
| `mass_1`, `mass_2`, ... | `(B,)` | All sampled waveform parameters, including derived quantities (`chirp_mass`, `mass_ratio`, `snr`, etc.). |

`L_wht = waveform_duration * effective_rate`, `L_raw = (waveform_duration + fduration) * effective_rate`, where `effective_rate = sample_rate / downsample_rate`.

The extra `fduration` seconds in the raw arrays is the FIR filter padding that the whitener consumes; it is retained in the raw arrays for downstream use.

---

## Changes since fork

The following describes divergences from the upstream `chreissel/GWDatasetGeneration` repo.

### `main.py`
- The return value of `injection()` was extended from `(whitened_injected, whitened_signal, params)` to `(whitened_injected, whitened_signal, raw_signal, whitened_bkg, raw_bkg, params)`.
- Saved HDF5 datasets renamed: old `injected_data`/`sig_only_data`/`bkg_only_data` → `whitened_injected`/`whitened_signal`/`whitened_bkg`/`raw_signal`/`raw_bkg`.
- Background is now saved separately as a whitened and raw channel rather than being derived as `injected - signal`.
- Added `--num-waveforms` CLI argument to override `num_waveforms` in the config without editing the file.
- Added `downsample_rate` support: all time-series are strided `[..., ::downsample_rate]` along the time axis before saving.
- Files are skipped if they already exist, enabling safe resumption of interrupted runs.

### `injections.py`
- Added `bkg_sample_rate` parameter (separate from `sample_rate`) so background files on disk can be at a different rate than the waveform generation rate.
- Added filtering of background files that are too short to fill the required window (`psd_length + fduration + waveform_duration`), preventing a `ValueError: high <= 0` from the dataloader.
- `injection()` now separately computes and returns `raw_signal`, `whitened_signal`, `raw_bkg`, and `whitened_bkg` in addition to `whitened_injected`.
- SNR reweighting distribution is now fully config-driven via `snr_reweighting.func` / `snr_reweighting.args` (no hardcoded `PowerLaw`).

### `waveforms.py`
- Added `BNS_IMRPhenomPv2` type: uses `IMRPhenomPv2` with full precessing spins. Spherical spin parameters `(a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl)` from the config are converted to Cartesian `(s1x, s1y, s1z, s2x, s2y, s2z)` automatically.
- `BNS_IMRPhenomPv2` enforces `m2 ≤ m1` by sorting the two sampled masses (element-wise `max`/`min`), allowing both to be drawn from the same distribution.
- The default `else` branch now uses `IMRPhenomPv2` instead of `IMRPhenomD`.
- Extrinsic projection parameters `dec`, `psi`, `phi` can optionally be specified in the `waveform` config block; if absent, isotropic defaults are used.
- Fixed a bug where `mass_2` was sampled as a 0-d scalar instead of a `(batch_size,)` tensor.

### `load_data.py`
- Removed dependency on the config file and hardcoded paths. `load_data()` now takes explicit arguments: `base_url`, `ifos`, `sample_rate`, `data_dir`.
- The minimum-duration filter on segments has been removed; all coincident network segments are downloaded regardless of length (short files are filtered at generation time in `injections.py`).
- The observing run (`O3a`, `O3b`, etc.) is now set in the `__main__` block via `base_url`.