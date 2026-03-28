# Gravitational Wave Dataset Generation — Detailed Notes

---

## 1. The Gravitational Wave Signal Model

A gravitational wave signal is a strain perturbation h(t) in spacetime. Ground-based detectors like LIGO (Hanford H1, Livingston L1) measure this strain as a dimensionless number — the fractional change in arm length, typically ~10⁻²¹ at merger. The signal from a compact binary coalescence (CBC) has two polarisations:

- **h₊ (plus polarisation)**: stretches space along one axis, compresses along the perpendicular.
- **h× (cross polarisation)**: same, but rotated 45 degrees.

What the detector actually measures is a linear combination of the two, weighted by the detector's antenna pattern functions F₊ and F×, which depend on the sky position and polarisation angle of the source relative to the detector orientation:

    h_detector(t) = F₊(α, δ, ψ) · h₊(t) + F×(α, δ, ψ) · h×(t)

---

## 2. Waveform Parameters

The waveform is fully described by a set of intrinsic and extrinsic parameters.

### 2.1 Intrinsic Parameters

These parameters determine the shape of the gravitational waveform in the source frame — they control the actual physics of the binary system.

**Component masses**

- **mass_1** (m₁): mass of the heavier companion, in solar masses (M☉).
- **mass_2** (m₂): mass of the lighter companion, in solar masses (M☉). Convention: m₁ ≥ m₂.

These are often reparametrised as:
- **Chirp mass** (ℳ): the combination (m₁m₂)^(3/5) / (m₁+m₂)^(1/5). This is the best-measured mass parameter from the waveform because it controls the rate of frequency evolution (the "chirp"). At leading post-Newtonian order, the frequency evolves as df/dt ∝ ℳ^(5/3), so ℳ is directly imprinted in the signal.
- **Mass ratio** (q): defined as q = m₂/m₁ ≤ 1. Together with ℳ, this uniquely determines m₁ and m₂.

For binary neutron stars (BNS), masses are bounded by:
- **Lower bound** (~1.0 M☉): below this the object would not be a neutron star formed from stellar evolution.
- **Upper bound** (~2.0–2.3 M☉): the Tolman–Oppenheimer–Volkoff (TOV) limit, above which neutron degeneracy pressure cannot support the star and it collapses to a black hole. Observationally constrained near 2.0 M☉ (PSR J0740+6620).

**Spin parameters**

Compact objects can be spinning. The spin angular momentum vector **S** = χ m² **ŝ** where χ ∈ [0, 1] is the dimensionless spin magnitude and **ŝ** is the unit spin direction.

For aligned-spin models (TaylorF2):
- **s1z, s2z**: the z-component of each spin, projected onto the orbital angular momentum axis. Positive = aligned with orbital angular momentum, negative = anti-aligned. Range: [−1, 1].

For precessing-spin models (IMRPhenomPv2), the full 3D spin vector of each body is specified. In the config we use spherical coordinates:
- **a_1, a_2**: spin magnitudes ∈ [0, 0.998] (the upper limit avoids the extremal Kerr singularity).
- **tilt_1, tilt_2**: polar tilt angle between the spin vector and the orbital angular momentum. tilt = 0 means aligned, tilt = π means anti-aligned, tilt = π/2 means in the orbital plane.
- **phi_12**: azimuthal angle between the two spin vectors in the plane perpendicular to the orbital angular momentum. Range: [0, 2π].
- **phi_jl**: azimuthal angle of the orbital angular momentum **L** around the total angular momentum **J**. Range: [0, 2π].

The conversion from these spherical coordinates to the Cartesian components required by IMRPhenomPv2 is:

    φ₁ = φ_jl
    φ₂ = φ_jl − φ₁₂
    s1x = a_1 · sin(tilt_1) · cos(φ₁)
    s1y = a_1 · sin(tilt_1) · sin(φ₁)
    s1z = a_1 · cos(tilt_1)
    s2x = a_2 · sin(tilt_2) · cos(φ₂)
    s2y = a_2 · sin(tilt_2) · sin(φ₂)
    s2z = a_2 · cos(tilt_2)

**Other intrinsic parameters**

- **phic**: orbital phase at coalescence, ∈ [0, 2π]. Determines the phase of the waveform at the moment of merger. Should be drawn from Uniform(0, 2π) for a representative dataset.
- **inclination** (ι): the angle between the binary's orbital angular momentum axis and the line of sight from Earth. ι = 0 (face-on): the binary's orbital plane is perpendicular to the line of sight; we see the system face-on, and the waveform is circularly polarised (maximum amplitude). ι = π/2 (edge-on): the orbital plane is along the line of sight; the waveform is linearly polarised (reduced amplitude by factor of 2).

### 2.2 Extrinsic Parameters

These parameters describe the location and orientation of the source relative to the detector network. They do not change the waveform polarisations, only the projection onto each detector.

- **distance** (D): luminosity distance in Mpc. The strain amplitude scales as h ∝ 1/D, so a source at twice the distance produces half the strain.
- **dec** (δ): declination, the celestial latitude. Combined with right ascension, this specifies the sky position. An isotropic distribution on the sky requires the Cosine prior: p(δ) ∝ cos(δ).
- **phi** (α): right ascension proxy, ∈ [−π, π]. Uniform prior gives isotropic sky coverage in the azimuthal direction.
- **psi** (ψ): polarisation angle, ∈ [0, π]. Describes the orientation of the binary's orbital plane projected on the sky. Uniform prior.

---

## 3. Waveform Approximants

Different approximants make different trade-offs between physical accuracy, spin coverage, and computational cost.

### TaylorF2 (used for type: BNS)

A frequency-domain post-Newtonian (PN) waveform model. It is derived by solving the equations of motion for the binary as a perturbative expansion in v/c, where v is the orbital velocity. Valid during the inspiral phase (before the bodies are close enough to merge). For BNS, the merger happens at high frequency (~1.5 kHz) and is outside the sensitive band, so TaylorF2 is accurate for the full observation window. Supports only aligned spins (s1z, s2z). Very fast to generate.

### IMRPhenomPv2 (used for type: BNS_IMRPhenomPv2, BBH)

A phenomenological waveform model that covers the full Inspiral–Merger–Ringdown (IMR) signal. "Phenom" refers to the fact that the model is fit to numerical relativity simulations. "Pv2" denotes the second version of the precessing-spin extension. Supports full 3D precessing spins. More computationally expensive than TaylorF2. Essential for BBH (where merger is in-band) and for studies of spin precession effects.

---

## 4. Prior Distributions

### 4.1 Training / Validation Prior (wide, uninformative)

The training prior is intentionally broader than the astrophysically realistic range. This ensures the model is exposed to the full parameter space and is not penalised for edge-case generalisations.

    mass_1 ~ Uniform(0.8, 2.8) M☉
    mass_2 ~ Uniform(0.8, 2.8) M☉   [then sorted: m1 = max(m1,m2), m2 = min(m1,m2)]

The sort enforces m₁ ≥ m₂ while keeping both marginal distributions uniform over [0.8, 2.8]. Note that the joint distribution is uniform over the triangle {0.8 ≤ m₂ ≤ m₁ ≤ 2.8}, so the individual marginals are triangular (not uniform), but sampling this way is the standard approach.

    a_1, a_2     ~ Uniform(0, 0.998)
    tilt_1, tilt_2  ~ Sine  [isotropic: p(tilt) ∝ sin(tilt)]
    phi_12, phi_jl  ~ Uniform(0, 2π)
    distance     ~ PowerLaw(100, 1000, α=2)   [uniform in volume: p(D) ∝ D²]
    phic         ~ Uniform(0, 2π)
    inclination  ~ Sine   [isotropic: p(ι) ∝ sin(ι)]
    dec          ~ Cosine  [isotropic sky: p(δ) ∝ cos(δ)]
    psi          ~ Uniform(0, π)
    phi (RA)     ~ Uniform(−π, π)
    SNR          ~ PowerLaw(5, 50)

### 4.2 Realistic / Test Prior

Used only for test sets. Designed to match the astrophysically motivated distribution used in GW parameter estimation (e.g. bilby):

    mass_1 ~ Triangular(min=1.0, max=2.5, mode=2.5)

The Triangular distribution with mode = max gives a prior that rises linearly from 1.0 M☉ and peaks at 2.5 M☉, mimicking the fact that more massive neutron stars are rarer but the distribution is not sharply peaked at a single value.

    mass_2 ~ Uniform(1.0, mass_1)   [conditional on mass_1]

This is the bilby ConditionalUniform: given a sampled m₁, m₂ is drawn uniformly from [1.0, m₁]. This automatically satisfies m₂ ≤ m₁ and gives a physically motivated joint distribution over the BNS mass plane.

The spin, distance, and angular parameters use the same distributions as the training prior.

### 4.3 Why not use the realistic prior for training?

If you train exclusively on the realistic prior, the model will be well-calibrated in the dense region (near m₁ ~ 2.5 M☉) but may fail on low-mass or equal-mass events that are rare in the training set. The wide uniform prior ensures uniform coverage of the parameter space during training. At evaluation time, the test prior tells you how the model performs on the distribution that real events actually follow.

---

## 5. Signal Injection Pipeline

### 5.1 Background data

Real detector noise is downloaded from the Gravitational Wave Open Science Center (GWOSC) using `load_data.py`. The files are stored in HDF5 format with one dataset per interferometer. For each batch of events, a window of duration

    window_length = psd_length + fduration + waveform_duration
                  = 64 + 2 + 64 = 130 seconds   (for the BNS config)

is randomly sampled from the background files. Files shorter than this window are excluded.

### 5.2 PSD estimation

The first `psd_length = 64 s` of the sampled window is used to estimate the one-sided Power Spectral Density S_n(f) using Welch's method: the segment is split into overlapping sub-windows, each is Fourier-transformed, and the results are averaged (using the median, which is robust to glitches). The frequency resolution is 1/fftlength = 0.5 Hz.

### 5.3 Whitening

The remaining `fduration + waveform_duration = 66 s` of background (the "kernel") is whitened by dividing its Fourier transform by the square root of the PSD:

    h̃_white(f) = h̃(f) / √S_n(f)

In the time domain this is implemented as convolution with a finite impulse response (FIR) filter of duration `fduration = 2 s`. Because the FIR filter introduces edge artefacts, `fduration/2 = 1 s` is cropped from each end after filtering, producing the final `waveform_duration = 64 s` whitened output. This is why the raw arrays are 66 s while the whitened arrays are 64 s.

After whitening, the noise should be statistically white (flat PSD) with unit variance per frequency bin. Deviation from flatness indicates spectral lines or non-stationarity.

### 5.4 SNR reweighting

A target SNR is drawn from the configured distribution (e.g. PowerLaw(5, 50)). The waveform is rescaled so that its network SNR (the quadrature sum of SNRs across detectors) matches this target. This is done before injection.

### 5.5 Injection

The rescaled waveform is added to the whitened background kernel at a fixed coalescence point (`right_pad = 1.0 s` from the right edge of the window). The resulting time series is then whitened.

### 5.6 Output datasets

For each event, five strain arrays are saved (shape: batch_size × n_ifos × L):

| Dataset | Length | Description |
|---|---|---|
| whitened_injected | 64 s | Signal + noise, whitened. Primary ML input. |
| whitened_signal | 64 s | Signal only, whitened. |
| whitened_bkg | 64 s | Noise only, whitened. |
| raw_signal | 66 s | Signal only, unwhitened (includes FIR padding). |
| raw_bkg | 66 s | Noise only, unwhitened (includes FIR padding). |

All arrays are downsampled by the stride `[..., ::downsample_rate]` before saving, giving an effective sample rate of `sample_rate / downsample_rate`. For the ai4gw@cern configs: 4096 / 16 = 256 Hz.

---

## 6. The Matched Filter SNR Formula

### 6.1 The formula

The optimal detection statistic for a known signal buried in Gaussian noise is the matched filter SNR. For detector j and template h_j, it is:

    ρ²_j = 4 ∫_{f_min}^{f_max} |h̃_j(f)|² / S_n^(j)(f)  df

where h̃_j(f) is the Fourier transform of the projected signal in detector j, and S_n^(j)(f) is the one-sided noise PSD of detector j. The network SNR is the quadrature sum over detectors:

    ρ_network = √( Σ_j ρ²_j )

### 6.2 Why the factor of 4?

It comes from two separate factors of 2 arising from PSD conventions.

**Factor of 2 from symmetry of the Fourier transform:**
The signal h(t) is real, so h̃(−f) = h̃*(f). The full two-sided integral from −∞ to +∞ therefore gets equal contributions from positive and negative frequencies:

    ∫_{-∞}^{∞} |h̃(f)|² / S_n^{two-sided}(f) df  =  2 ∫_0^{∞} |h̃(f)|² / S_n^{two-sided}(f) df

**Factor of 2 from the one-sided PSD convention:**
The one-sided PSD is defined to contain all the noise power in positive frequencies only:

    S_n^{one-sided}(f) = 2 × S_n^{two-sided}(f)    for f > 0

Combining both:

    ∫_{-∞}^{∞} |h̃|² / S_n^{two-sided} df
      = 2 ∫_0^{∞} |h̃|² / S_n^{two-sided} df
      = 2 ∫_0^{∞} |h̃|² / (S_n^{one-sided}/2) df
      = 4 ∫_0^{∞} |h̃|² / S_n^{one-sided} df

So the factor of 4 is a convention artifact: it ensures that ρ² computed with the one-sided PSD equals the optimal signal-to-noise ratio of the full two-sided matched filter.

### 6.3 Why can SNR be much larger than 1?

A naive reading of the formula might suggest ρ ≈ 1 if the signal is "at the noise level." But this misses the key point: the integral accumulates coherently over all frequencies in the band. For a BNS sweeping from 20 Hz to ~1000 Hz, that is roughly 1000 Hz of bandwidth, sampled at Δf = 0.5 Hz, giving ~2000 frequency bins.

At any single frequency bin, the signal is far below the noise — the signal is literally invisible in the raw data. But in each bin the ratio |h̃(f)|²/S_n(f) is a small positive number, and the integral sums ~2000 such contributions. The matched filter is the optimal linear filter precisely because it uses the known phase evolution of the waveform to add these contributions coherently (in phase), whereas the noise contributions are incoherent and average down.

The result is an SNR that accumulates roughly as:

    ρ² ≈ (signal amplitude)² × (number of cycles in band) / (noise level per cycle)

For a BNS at ~200 Mpc with O3 sensitivity (ASD ~ 5×10⁻²⁴ strain/√Hz at 100 Hz), the integral gives ρ ~ 5–10. At ~40 Mpc, ρ ~ 50. This is consistent with the training prior SNR range of [5, 50].

A useful intuition: matched filtering is equivalent to correlating the data with a copy of the expected signal. Over N cycles, the signal correlation grows as N while the noise correlation grows as √N (random walk), so the SNR scales as √N. More cycles = higher SNR, which is why long BNS inspirals (minutes in band) can be detected at lower amplitude than short BBH mergers (seconds in band).
