"""Converting between waveform and time-frequency representations of audio data."""

import sys
sys.path.append('../..')

import warnings
import numpy as np
import torch
from edm2.torch_utils import persistence
from edm2.torch_utils import misc
from torchaudio import transforms


@persistence.persistent_class
class MelSpectrogramEncoder:
    def __init__(
        self,
        sample_rate = 16000,
        hop_length = 256,
        win_length = 1024,
        n_fft = 1024,
        n_mels = 80,
        f_min = 0.0,
        f_max = 8000,
        compression = True,
        raw_mean = -7.98, # Computed on training set
        raw_std = 2.43,  # Computed on training set
        final_mean = 0,
        final_std = 0.5
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.compression = compression

        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale

        self.audio_to_mel = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1,
            norm='slaney',
            mel_scale='slaney',
        )

    def encode(self, x): 
        self.audio_to_mel = self.audio_to_mel.to(x.device)
        x = self.audio_to_mel(x)
        if self.compression:
            x = torch.log(torch.clamp(x, min=1e-5))
        x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        return x
