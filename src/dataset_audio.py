import os
import numpy as np
import torch
from glob import glob
import soundfile as sf
from numpy import random

import edm2.torch_utils.distributed as dist

#----------------------------------------------------------------------------
# Dataset class that loads audio recursively from the specified directory

class AudioFolderDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                     # Path to directory or zip.
        segment_size  = None,     # Segment size for audio.
        deterministic = False,    # Use deterministic random number generator?
        normalize     = False,    # Normalize audio to [-1, 1]?
        max_size      = None,     # Artificially limit the size of the dataset. None = no limit.
        random_seed   = 0,        # Random seed to use when applying max_size.
        cache         = False,    # Cache images in CPU memory?
    ):
        self.path = path
        self.segment_size = segment_size
        self.deterministic = deterministic
        self.normalize = normalize
        self.rng = random.default_rng()

        if os.path.isdir(self.path):
            self.all_fnames = sorted(glob(os.path.join(self.path, "**", "*.wav"), recursive=True))
            with open('all_fnames.txt', 'w') as f:
                for item in self.all_fnames:
                    f.write("%s\n" % item) # print all_fnames to file
        elif isinstance(self.path, (tuple, list)):
            self.all_fnames = []
            for path in self.path:
                assert os.path.isdir(path), f"Invalid directory: {path}"
                self.all_fnames.extend(sorted(glob(os.path.join(path, "**", "*.wav"), recursive=True)))  
        else:
            raise IOError('Path must point to a directory')

        supported_ext = {'.wav', '.flac', '.mp3'}
        self.audio_fnames = sorted(fname for fname in self.all_fnames \
                                   if os.path.splitext(fname)[-1] in supported_ext)
        if len(self.audio_fnames) == 0:
            raise IOError('No audio files found in the specified path')

        self.raw_shape = [len(self.audio_fnames)] + list(self.load_raw_audio(0).shape) # (NCHW).
        self.cache = cache
        self.cached_audio = dict() # {raw_idx: np.ndarray, ...}
        self.raw_labels = None
        self.label_shape = None

        # Apply max_size.
        self.raw_idx = np.arange(self.raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self.raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self.raw_idx)
            self.raw_idx = np.sort(self.raw_idx[:max_size])

    def __len__(self):
        return self.raw_idx.size

    def load_raw_audio(self, raw_idx):
        fname = self.audio_fnames[raw_idx]
        audio, _ = sf.read(fname, dtype='float32', always_2d=True)
        audio = audio.T
        if self.normalize:
            audio = audio / np.abs(audio).max()

        rng = random.default_rng(raw_idx) if self.deterministic else self.rng
        if self.segment_size is not None:
            if audio.shape[-1] >= self.segment_size:
                max_audio_start = audio.shape[-1] - self.segment_size
                audio_start = rng.integers(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = np.pad(audio, ((0, 0), (0, self.segment_size - audio.shape[-1])))
        return audio

    def __getitem__(self, idx):

        raw_idx = self.raw_idx[idx]
        audio = self.cached_audio.get(raw_idx, None)
        if audio is None:
            audio = self.load_raw_audio(raw_idx)
            if self.cache:
                self.cached_audio[raw_idx] = audio
        assert isinstance(audio, np.ndarray)

        return audio
