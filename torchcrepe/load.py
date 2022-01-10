import os

import numpy as np
import torch
import torchcrepe
from scipy.io import wavfile


def audio(filename):
    """Load audio from disk"""
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    # PyTorch is not compatible with non-writeable arrays, so we make a copy
    return torch.tensor(np.copy(audio))[None], sample_rate


def model(device, capacity='full', cl43=False, overwrite_capacity=None):
    """Preloads model from disk"""
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.model = torchcrepe.Crepe(capacity if overwrite_capacity is None else overwrite_capacity)

    # Load weights
    folder = 'assets'
    if cl43:
        folder = 'assets/cl43'
    file = os.path.join(os.path.dirname(__file__), folder, f'{capacity}.pth')
    torchcrepe.infer.model.load_state_dict(
        torch.load(file, map_location=device))

    # Place on device
    torchcrepe.infer.model = torchcrepe.infer.model.to(torch.device(device))

    # Eval mode
    torchcrepe.infer.model.eval()
