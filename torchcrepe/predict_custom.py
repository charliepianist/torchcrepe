import torchcrepe
import torch
import resampy
import librosa
import numpy as np
import os
from scipy.io import wavfile

def load_audio(filename):
    """Load audio from disk"""
    # audio, sample_rate = librosa.load(filename, sr=torchcrepe.SAMPLE_RATE)
    sample_rate, audio = wavfile.read(filename)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    
    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # PyTorch is not compatible with non-writeable arrays, so we make a copy
    return torch.tensor(np.copy(audio))[None], sample_rate

def predict(audio,
            sample_rate,
            hop_length=None,
            fmin=50.,
            fmax=torchcrepe.MAX_FMAX,
            model='base_tiny',
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=False,
            batch_size=None,
            device='cpu',
            pad=True,
            capacity='tiny',
            special=None):
    """Performs pitch estimation

    Arguments
        audio (torch.tensor [shape=(1, time)])
            The audio signal
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'base_tiny', .
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_periodicity (bool)
            Whether to also return the network confidence
        batch_size (int)
            The number of frames per batch
        device (string)
            The device used to run inference
        pad (bool)
            Whether to zero-pad the audio

    Returns
        pitch (torch.tensor [shape=(1, 1 + int(time // hop_length))])
        (Optional) periodicity (torch.tensor
                                [shape=(1, 1 + int(time // hop_length))])
    """

    results = []

    # Postprocessing breaks gradients, so just don't compute them
    with torch.no_grad():

        # Preprocess audio
        generator = preprocess(audio,
                                    sample_rate,
                                    hop_length,
                                    batch_size,
                                    device,
                                    pad)
        for frames in generator:
        #     for temp in range(len(frames)):
        #         print(infer(frames.index_select(0, torch.LongTensor([temp])), device, model))
            # Infer independent probabilities for each pitch bin
            probabilities = infer(frames, device, model, capacity=capacity, special=special)
            # print(probabilities[0])
            # print(probabilities[1])
            # print(probabilities[2])
            # for temp in range(1000, 1100):
            #     print(probabilities[temp])
            # for temp in range(2000, 2100):
            #     print(probabilities[temp])
            # shape=(batch, 360, time / hop_length)
            # probabilities = probabilities.reshape(
            #     probabilities.size(0), -1, torchcrepe.PITCH_BINS)#.transpose(1, 2)
            # print(probabilities.size(0), -1, torchcrepe.PITCH_BINS)
            probabilities = probabilities.reshape(
                -1, probabilities.size(0), torchcrepe.PITCH_BINS)
            # print(probabilities)
            # print(probabilities.transpose(1, 2))
            probabilities = probabilities.transpose(1, 2)
            # Convert probabilities to F0 and periodicity
            result = torchcrepe.postprocess(probabilities,
                                 fmin,
                                 fmax,
                                 decoder,
                                 return_periodicity)

            # Place on same device as audio to allow very long inputs
            if isinstance(result, tuple):
                result = (result[0].to(audio.device),
                          result[1].to(audio.device))
            else:
                 result = result.to(audio.device)

            results.append(result)

    # Time multiplier
    time_mult = 0.01
    if hop_length != None:
        time_mult = hop_length / sample_rate

    # Split pitch and periodicity
    if return_periodicity:
        pitch, periodicity = zip(*results)
        # 10ms timestamps
        return [i*time_mult for i in range(len(torch.cat(pitch, 1)[0]))], torch.cat(pitch, 1)[0], torch.cat(periodicity, 1), []

    # Concatenate
    return [i*time_mult for i in range(len(torch.cat(results, 1)[0]))], torch.cat(results, 1)[0], [], []

def infer(frames, device='cpu', model='base_tiny', embed=False, capacity='tiny', special=None):
    """Forward pass through the model

    Arguments
        frames (torch.tensor [shape=(time / hop_length, 1024)])
            The network input
        model (string)
            The model capacity. One of 'base_tiny'.
        embed (bool)
            Whether to stop inference at the intermediate embedding layer

    Returns
        logits (torch.tensor [shape=(1 + int(time // hop_length), 360)]) OR
        embedding (torch.tensor [shape=(1 + int(time // hop_length),
                                       embedding_size)])
    """
    # Load the model if necessary
    if not hasattr(torchcrepe.infer, 'model') or not hasattr(torchcrepe.infer, 'capacity') or \
       (hasattr(torchcrepe.infer, 'capacity') and torchcrepe.infer.capacity != model):
        load_model(device, model, cl43=True, overwrite_capacity=capacity, special=special)

    # Move model to correct device (no-op if devices are the same)
    torchcrepe.infer.model = torchcrepe.infer.model.to(device)

    # Apply model
    return torchcrepe.infer.model(frames, embed=embed)


def load_model(device, capacity='full', cl43=False, overwrite_capacity=None, special=None):
    """Preloads model from disk"""
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.model = torchcrepe.Crepe(capacity if overwrite_capacity is None else overwrite_capacity)
    if special == 'KL':
        torchcrepe.infer.model = torchcrepe.CrepeKL(capacity if overwrite_capacity is None else overwrite_capacity)
    elif special == 'Small':
        torchcrepe.infer.model = torchcrepe.CrepeSmall(capacity if overwrite_capacity is None else overwrite_capacity)
    elif special == 'Large':
        torchcrepe.infer.model = torchcrepe.CrepeLarge(capacity if overwrite_capacity is None else overwrite_capacity)

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



def preprocess(audio,
               sample_rate,
               hop_length=None,
               batch_size=None,
               device='cpu',
               pad=True):
    """Convert audio to model input

    Arguments
        audio (torch.tensor [shape=(1, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        batch_size (int)
            The number of frames per batch
        device (string)
            The device to run inference on
        pad (bool)
            Whether to zero-pad the audio

    Returns
        frames (torch.tensor [shape=(1 + int(time // hop_length), 1024)])
    """
    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample
    if sample_rate != torchcrepe.SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * torchcrepe.SAMPLE_RATE / sample_rate)

    # Get total number of frames

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.size(1) // hop_length)
        audio = torch.nn.functional.pad(
            audio,
            (torchcrepe.WINDOW_SIZE // 2, torchcrepe.WINDOW_SIZE // 2))
    else:
        total_frames = 1 + int((audio.size(1) - torchcrepe.WINDOW_SIZE) // hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size
    
    # Generate batches
    for i in range(0, total_frames, batch_size):

        # Batch indices
        start = max(0, i * hop_length)
        end = min(audio.size(1),
                  (i + batch_size - 1) * hop_length + torchcrepe.WINDOW_SIZE)
        
        # Chunk
        frames = torch.nn.functional.unfold(
            audio[:, None, None, start:end],
            kernel_size=(1, torchcrepe.WINDOW_SIZE),
            stride=(1, hop_length))

        # shape=(1 + int(time / hop_length, 1024)
        frames = frames.transpose(1, 2).reshape(-1, torchcrepe.WINDOW_SIZE)

        # Place on device
        frames = frames.to(device)

        # Mean-center
        frames -= frames.mean(dim=1, keepdim=True)

        # Scale
        # Note: during silent frames, this produces very large values. But
        # this seems to be what the network expects.
        frames /= torch.max(torch.tensor(1e-10, device=frames.device),
                            frames.std(dim=1, keepdim=True))

        yield frames

def resample(audio, sample_rate):
    """Resample audio"""
    # Store device for later placement
    device = audio.device

    # Convert to numpy
    audio = audio.detach().cpu().numpy().squeeze(0)
    audio = audio.T

    # Resample
    # We have to use resampy if we want numbers to match Crepe
    audio = resampy.resample(audio, sample_rate, torchcrepe.SAMPLE_RATE)

    # Convert to pytorch
    return torch.tensor(audio, device=device).unsqueeze(0)