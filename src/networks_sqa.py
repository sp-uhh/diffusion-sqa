# Adapted from https://github.com/NVlabs/edm2/blob/main/training/networks_edm2.py.

"""
Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models", with adaptations
to reduce the network size by default, remove class conditioning and making attention
setup independent of resolution, specified by the block index instead.
"""

import sys
sys.path.append('../..')

import torch
from edm2.torch_utils import persistence
from edm2.training.networks_edm2 import mp_silu, mp_cat, MPFourier, MPConv, Block


#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        audio_channels,                     # Audio channels.
        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,4],      # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 1,            # Number of residual blocks per resolution.
        attn_levels         = [2,],         # List of block indices with self-attention.
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = audio_channels + 1
        for level, channels in enumerate(cblock):
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'lvl{level}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'lvl{level}_down'] = Block(cout, cout, cemb, flavor='enc', 
                                                     resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'lvl{level}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', 
                                                           attention=(level in attn_levels), 
                                                           **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            if level == len(cblock) - 1:
                self.dec[f'lvl{level}_in0'] = Block(cout, cout, cemb, flavor='dec', 
                                                    attention=True, **block_kwargs)
                self.dec[f'lvl{level}_in1'] = Block(cout, cout, cemb, flavor='dec', 
                                                    **block_kwargs)
            else:
                self.dec[f'lvl{level}_up'] = Block(cout, cout, cemb, flavor='dec', 
                                                   resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'lvl{level}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', 
                                                           attention=(level in attn_levels), 
                                                           **block_kwargs)
        self.out_conv = MPConv(cout, audio_channels, kernel=[3,3])

    def forward(self, x, noise_labels):
        # Embedding.
        emb = mp_silu(self.emb_noise(self.emb_fourier(noise_labels)))

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x

#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

@persistence.persistent_class
class Precond(torch.nn.Module):
    def __init__(self,
        freq_resolution,        # Frequency resolution.
        audio_channels,         # Audio channels.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.freq_resolution = freq_resolution
        self.audio_channels = audio_channels
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = UNet(audio_channels=audio_channels, **unet_kwargs)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, force_fp32=False, return_logvar=False, **kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, c_noise)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x

#----------------------------------------------------------------------------
