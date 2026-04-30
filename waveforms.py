import torch
import importlib
from ml4gw.distributions import Cosine
from torch.distributions import Uniform
from ml4gw.waveforms import IMRPhenomPv2, TaylorF2 # IMRPhenomPv2 uses full 3D
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from ml4gw.gw import get_ifo_geometry, compute_observed_strain
from utils import load_config

def _rejection_sample_masses(mc_dist, q_dist, m1_min, m1_max, m2_min, m2_max, batch_size, device):
    """
    Sample (chirp_mass, mass_ratio) uniformly and reject pairs whose component
    masses fall outside [m1_min, m1_max] x [m2_min, m2_max]. Loops until
    exactly batch_size valid samples are collected.
    """
    accepted_mc, accepted_q, accepted_m1, accepted_m2 = [], [], [], []
    remaining = batch_size

    while remaining > 0:
        n = max(4 * remaining, batch_size)
        mc = mc_dist.sample((n,)).to(device)
        q  = q_dist.sample((n,)).to(device)
        m1, m2 = chirp_mass_and_mass_ratio_to_components(mc, q)

        mask = (m1 >= m1_min) & (m1 <= m1_max) & (m2 >= m2_min) & (m2 <= m2_max)
        accepted_mc.append(mc[mask])
        accepted_q.append(q[mask])
        accepted_m1.append(m1[mask])
        accepted_m2.append(m2[mask])
        remaining -= int(mask.sum().item())

    mc = torch.cat(accepted_mc)[:batch_size]
    q  = torch.cat(accepted_q)[:batch_size]
    m1 = torch.cat(accepted_m1)[:batch_size]
    m2 = torch.cat(accepted_m2)[:batch_size]
    return mc, q, m1, m2


def generate_signals(config, device: str, save: bool):
    
    waveform_duration = config.general.waveform_duration
    batch_size = config.general.batch_size
    sample_rate = config.general.sample_rate
    ifos = config.general.ifos
    f_min = config.general.f_min
    f_max = config.general.f_max
    f_ref = config.general.f_ref
    waveform_dict = config.waveform
    right_pad = config.general.right_pad

    # nyquist = sample_rate / 2
    # num_samples = int(waveform_duration * sample_rate)
    # num_freqs = num_samples // 2 + 1

    # frequencies = torch.linspace(0, nyquist, num_freqs).to(device)
    # freq_mask = (frequencies >= f_min) * (frequencies < f_max).to(device)

    param_dict = {}
    attrs = [x for x in dir(waveform_dict) if '__' not in x]
    params = {}
    for k in attrs:
        attrs_config = getattr(waveform_dict, k)
        func_path = getattr(attrs_config, 'func')

        module_name, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        if 'args' in dir(attrs_config):
            args = getattr(attrs_config, 'args')
            args = [a if not isinstance(a, str) else params[a] for a in args]
            param_dict[k] = func(*args)
        else:
            param_dict[k] = func()

        # if the distribution was built with tensor args it already has batch_shape
        # (e.g. Uniform(1.0, mass_1_tensor)), so just call .sample() to avoid
        # producing shape (batch_size, batch_size)
        if param_dict[k].batch_shape == torch.Size([batch_size]):
            params[k] = param_dict[k].sample().to(device)
        else:
            params[k] = param_dict[k].sample((batch_size,)).to(device)

    if config.general.type=='BNS':
        approximant = TaylorF2().to(device)

        # get correct parameters
        q = params['mass_2']/params['mass_1']
        params['chirp_mass'] = (q/(1+q)**2)**(3/5.)*(params['mass_2']+params['mass_1'])
        params['mass_ratio'] = q
        params["chi1"], params["chi2"] = params["s1z"], params["s2z"] # ???

    elif config.general.type=='BNS_IMRPhenomPv2':
        approximant = IMRPhenomPv2().to(device)

        if 'chirp_mass' in params and 'mass_ratio' in params:
            if getattr(config.general, 'rejection_sampling', False):
                m1_bounds = config.general.m1_bounds
                m2_bounds = config.general.m2_bounds
                params['chirp_mass'], params['mass_ratio'], params['mass_1'], params['mass_2'] = \
                    _rejection_sample_masses(
                        param_dict['chirp_mass'], param_dict['mass_ratio'],
                        m1_bounds[0], m1_bounds[1],
                        m2_bounds[0], m2_bounds[1],
                        batch_size, device,
                    )
            else:
                params['mass_1'], params['mass_2'] = chirp_mass_and_mass_ratio_to_components(
                    params['chirp_mass'], params['mass_ratio']
                )
        else:
            # enforce m2 <= m1 by sorting
            m1 = torch.max(params['mass_1'], params['mass_2'])
            m2 = torch.min(params['mass_1'], params['mass_2'])
            params['mass_1'], params['mass_2'] = m1, m2

            q = params['mass_2'] / params['mass_1']
            params['chirp_mass'] = (q / (1 + q)**2)**(3/5.) * (params['mass_2'] + params['mass_1'])
            params['mass_ratio'] = q

        # convert spherical spin parameters to Cartesian components
        # phi_12 = phi_1 - phi_2, phi_jl sets the azimuthal frame
        phi_1 = params['phi_jl']
        phi_2 = params['phi_jl'] - params['phi_12']
        params['s1x'] = params['a_1'] * torch.sin(params['tilt_1']) * torch.cos(phi_1)
        params['s1y'] = params['a_1'] * torch.sin(params['tilt_1']) * torch.sin(phi_1)
        params['s1z'] = params['a_1'] * torch.cos(params['tilt_1'])
        params['s2x'] = params['a_2'] * torch.sin(params['tilt_2']) * torch.cos(phi_2)
        params['s2y'] = params['a_2'] * torch.sin(params['tilt_2']) * torch.sin(phi_2)
        params['s2z'] = params['a_2'] * torch.cos(params['tilt_2'])

    else:
        approximant = IMRPhenomPv2().to(device)

        params["mass_1"], params["mass_2"] = chirp_mass_and_mass_ratio_to_components(
            params["chirp_mass"], params["mass_ratio"]
        )
        params["s1z"], params["s2z"] = params["chi1"], params["chi2"]


    waveform_generator = TimeDomainCBCWaveformGenerator(
        approximant=approximant,
        sample_rate=sample_rate,
        f_min=f_min,
        duration=waveform_duration,
        right_pad=right_pad,
        f_ref=f_ref,
    ).to(device)

    hc, hp = waveform_generator(**params)

    # Waveform projection: use config-defined priors if provided, else defaults
    # dec ~ Cosine (isotropic sky), psi ~ Uniform(0, pi), phi/RA ~ Uniform(-pi, pi)
    if 'dec' not in params:
        params['dec'] = Cosine().sample((batch_size,)).to(device)
    if 'psi' not in params:
        params['psi'] = Uniform(0, torch.pi).sample((batch_size,)).to(device)
    if 'phi' not in params:
        params['phi'] = Uniform(-torch.pi, torch.pi).sample((batch_size,)).to(device)

    tensors, vertices = get_ifo_geometry(*ifos)

    waveforms = compute_observed_strain(
        dec=params['dec'],
        psi=params['psi'],
        phi=params['phi'],
        detector_tensors=tensors.to(device),
        detector_vertices=vertices.to(device),
        sample_rate=sample_rate,
        cross=hc,
        plus=hp,
    )
    if save:
        return True
    else:
        return waveforms, params

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path='config.yaml')

    generate_signals(config, device=device, save=True)
