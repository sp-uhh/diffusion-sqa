# Adapted from https://github.com/NVlabs/edm2/blob/main/training/training_loop.py.

"""EDM2 loss function."""

import sys
sys.path.append('../..')

import torch
from edm2.torch_utils import persistence


#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, data):
        rnd_normal = torch.randn([data.shape[0], 1, 1, 1], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(data) * sigma
        denoised, logvar = net(data + noise, sigma, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - data) ** 2) + logvar
        return loss
