# Adapted from https://github.com/NVlabs/edm2/blob/main/generate_images.py.

import sys
sys.path.append('./edm2')

import os
import pickle
import click
from tqdm import tqdm
from glob import glob
from more_itertools import unique_everseen

import librosa
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F

from edm2 import dnnlib
from edm2.torch_utils import distributed as dist

import warnings
warnings.filterwarnings('ignore', 'No device id is provided via `init_process_group` or `barrier `.')

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randint(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(size=size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randint_like(self, input, **kwargs):
        return self.randint(input.shape, dtype=input.dtype, layout=input.layout, device=input.device, **kwargs)

class AudioIterable:
    def __init__(self, data_path, encoder, device='cuda'):

        self.device = device
        self.encoder = encoder
        candidate_files = sorted(glob(f'{data_path}/**/*.wav', recursive=True))
        if len(candidate_files) % dist.get_world_size():
            # Extend the files list (repeating the last element) to be divisible by the number of GPUs (duplicates are removed in the end)
            dist.print0(f'Extending the list of files by {len(candidate_files) % dist.get_world_size()} to make divisible by {dist.get_world_size()}')
            candidate_files = candidate_files + [candidate_files[-1]] * (dist.get_world_size() - len(candidate_files) % dist.get_world_size())
      
        self.candidate_files = candidate_files[dist.get_rank() :: dist.get_world_size()]
        dist.print0(f'Evaluating {len(self.candidate_files)} audio files per rank...')

    def __len__(self):
        return len(self.candidate_files)

    def __iter__(self):
        for audio_path in self.candidate_files:

            audio_id = os.path.splitext(os.path.basename(audio_path))[0]
            audio_np, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            if sr != self.encoder.sample_rate:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.encoder.sample_rate)
            audio = torch.as_tensor(audio_np.T).to(self.device).unsqueeze(0)

            r = dnnlib.EasyDict(audio=None, audio_raw=None, labels=None, seeds=None, audio_id=audio_id)
            r.audio_raw = audio_np.squeeze()
            r.audio = self.encoder.encode(audio)

            yield r

# Grathwohl et al.
def hutchinson_trace(x_out, x_in, noise=None):
    e_dzdx = torch.autograd.grad(x_out, x_in, noise)[0]
    e_dzdx_e = e_dzdx * noise
    approx_tr_dzdx = e_dzdx_e.view(x_in.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

def likelihood_edm_sampler(
    net, data, num_steps=32, second_order=True, seed=0, eps=1e-5, 
    sigma_min=0.002, sigma_max=80, rho=7, dtype=torch.float32, 
):

    def denoise(x, t):
        target_dim = 8
        padding = target_dim - x.shape[-1] % target_dim
        x = F.pad(x, (0, padding))

        Dx = net(x, t).to(dtype)
        if padding > 0:
            Dx = Dx[...,:-padding]
        return Dx
    
    def drift_fn(x, t):
        return (x - denoise(x, t)) / t
    
    rnd = StackedRandomGenerator(data.device, (seed,))
    trace_noise = rnd.randint_like(data, low=0, high=2).float() * 2 - 1

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=data.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, eps*torch.ones_like(t_steps[:1])]) # t_N = 0
    t_steps = t_steps.flip(0) # forward process

    # Main sampling loop.
    x_next = data.to(dtype)
    d_logp_next = torch.zeros(data.shape[0], device=data.device, dtype=dtype)

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):

        x_cur = x_next
        d_logp_cur = d_logp_next

        t_hat = t_cur
        x_hat = x_cur

        # Euler step.
        with torch.enable_grad():
            x_hat.requires_grad_(True)
            d_cur = drift_fn(x_hat, t_hat)
            tr_jac_cur = hutchinson_trace(d_cur, x_hat, noise=trace_noise)
        x_hat = x_hat.detach()

        x_next = x_hat + (t_next - t_hat) * d_cur
        d_logp_next = d_logp_cur + (t_next - t_hat) * tr_jac_cur

        # Apply 2nd order correction.
        if second_order:
            with torch.enable_grad():
                x_next.requires_grad_(True)
                d_prime = drift_fn(x_next, t_next)
                tr_jac_prime = hutchinson_trace(d_prime, x_next, noise=trace_noise)
            x_next = x_next.detach()

            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            d_logp_next = d_logp_cur + (t_next - t_hat) * (0.5 * tr_jac_cur + 0.5 * tr_jac_prime)

    return x_next, d_logp_next

def calculate_logprior(z, sigma=80):    
    """Compute the log of the pdf of a multivariate Gaussian with mean 0 and standard deviation `sigma`."""
    N = np.prod(z.shape[1:])
    logprior = -N / 2. * np.log(2 * np.pi * sigma ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * sigma ** 2)
    return logprior, N

def calculate_loglikelihood(net, data, num_steps=32, seed=0):

    with torch.no_grad():
        x_T, delta_logp = likelihood_edm_sampler(
            net=net, 
            data=data,
            num_steps=num_steps,
            seed=seed
        )

        prior_logp, N = calculate_logprior(x_T)
        logp = (delta_logp + prior_logp) / N
    
    return logp

    
#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option("--checkpoint",  help='Path to the model snapshot or reconstructed EMA pickle file.',    type=str, required=True)
@click.option("--data_dir",    help='Path to the directory containing the wav files to be evaluated.', type=str, required=True)
@click.option("--output_file", help='Where to save the results. Typically a path to a csv or txt.',    type=str, required=True)
@click.option("--num_steps",   help='Number of steps for the solver.',                                 type=int, default=32, show_default=True)
@click.option('--seed',        help='Seed for noise generation.',                                      type=int, default=0, show_default=True)
@click.option("--device",      help='Device where the calculations will be computed (cuda or cpu).',   type=click.Choice(['cuda', 'cpu']), default="cuda")

def cmdline(**opts):
    """Calculate the diffusion log likelihoods using a given checkpoint.

    Examples:

    \b
    # Calculate the diffusion log likelihoods on the EARS-WHAM noisy test set using a single gpu and save them in a csv file
    python calculate_likelihood.py --checkpoint=ckpt/phema-0037748-0.080.pkl \\
        --data_dir=./data/EARS-WHAM-16k/test/noisy --output_file=./results/ears-wham-noisy.csv

    \b
    # Calculate the diffusion log likelihoods on the EARS-WHAM noisy test set using 4 gpus and save them in a csv file
    torchrun --standalone --nproc_per_node=4 calculate_likelihood.py --checkpoint=ckpt/phema-0037748-0.080.pkl \\
        --data_dir=./data/EARS-WHAM-16k/test/noisy --output_file=./results/ears-wham-noisy.csv
    """
    opts = dnnlib.EasyDict(opts)

    dist.init()
    dist.print0(f"Results will be saved in {opts.output_file}")
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0(f'Loading network from {opts.checkpoint} ...')
    with open(opts.checkpoint, 'rb') as f:
        data = pickle.load(f)

    net = data['ema'].to(opts.device)
    assert net is not None

    encoder = data.get('encoder', None)
    dist.print0(f'Setting up {type(encoder).__name__}...')
    if encoder is None:
        dist.print0("Encoder not found in the checkpoint. Using the default MelSpectrogramEncoder.")
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.MelSpectrogramEncoder', device=opts.device)

    if dist.get_rank() == 0:
        # Create a csv file to store the results.
        with open(opts.output_file, 'w') as f:
            f.write('audio_id,logp\n')
        torch.distributed.barrier()

    audio_iter = AudioIterable(opts.data_dir, encoder=encoder, device=opts.device)
    
    for r in tqdm(audio_iter, unit='file', disable=(dist.get_rank() != 0)):

        logp = calculate_loglikelihood(net, r.audio, num_steps=opts.num_steps, seed=opts.seed)

        # Append the results to the csv file.
        with open(opts.output_file, 'a') as f:
            f.write(f'{r.audio_id},{round(logp.item(), 5)}\n')

    torch.distributed.barrier()

    if dist.get_rank() == 0:
        with open(opts.output_file, 'r') as f:
            data = f.readlines()
        with open(opts.output_file, 'w') as f:
            f.writelines(unique_everseen(data))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
