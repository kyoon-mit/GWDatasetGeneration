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