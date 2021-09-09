import os
import pickle
import bisect
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile

import torchcrepe
from torchcrepe.predict_custom import load_audio

DATA_DIR = '/home/azureuser/cloudfiles/code/Users/cl43/torchcrepe/torchcrepe/data/MIR-QBSH/waveFile'
DATA_DIR2 = '/home/azureuser/cloudfiles/code/Users/cl43/torchcrepe/torchcrepe/data/MIR-QBSH-copy/waveFile'
CACHE_FILE = '/home/azureuser/cloudfiles/code/Users/cl43/torchcrepe/torchcrepe/data/mir_cache.pkl'
CACHE_FILE2 = '/home/azureuser/cloudfiles/code/Users/cl43/torchcrepe/torchcrepe/data/mir_cache-copy.pkl'
HOP_LENGTH_MS = 5 # Note that the target outputs are every 256 frames, or approximately every 31.25 ms
HOP_LENGTH = 256 # Ignore HOP_LENGTH_MS if this is not None
VALIDATION_SPLIT = 0.2 # How much of the data to be validation
TEST_SPLIT = 0.2 # How much of the data to be test

MIN_HZ = 32.75 # Minimum output frequency

def save_obj(filename, objs):
    with open(filename, 'wb') as outp:
        for obj in objs:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_obj(filename, num_to_read):
    objs = []
    with open(filename, 'rb') as inp:
        for i in range(num_to_read):
            objs.append(pickle.load(inp))
        return objs

class MirDataset(Dataset):
    # Existing_dataset allows copying all the information over from a different dataset so we don't have to reload everything for the validation dataset
    # cloned = use copy of dataset (in case reading the same dataset concurrently is bad)
    # cache = Use cached values for paths, end_indices, mapped_indices (saves a lot of time)
    # existing_dataset = copy values from an existing instance of MirDataset
    # strong_penalty = targets as [0, ..., 0, 1, 0, ..., 0] if True or as Gaussian blurred vectors if False
    def __init__(self, mini=False, split='train', existing_dataset=None, cache=True, cloned=False, strong_penalty=False):
        data_dir = DATA_DIR2 if cloned else DATA_DIR
        cache_file = CACHE_FILE2 if cloned else CACHE_FILE
        self.split = split
        self.strong_penalty = strong_penalty
        if existing_dataset is None:
            if cache and os.path.exists(cache_file):
                arr = read_obj(cache_file, 3)
                self.paths = arr[0]
                self.end_indices = arr[1]
                self.mapped_indices = arr[2]
            else:
                self.paths = [] 
                self.end_indices = [] # Last index (exclusive) for each path, in terms of number of data points
                self.mapped_indices = [] # 2D array of shape (len(paths), <variable length>) of indices within each path that are nonzero pitches
                curr_idx = 0
                # Get all files for which a pitch vector is present
                for root, dirs, files in os.walk(data_dir):
                    for name in files:
                        if name[-3:] == '.pv':
                            mapped_idx = []

                            # Update path, read in pitch vector to get the length
                            self.paths.append(os.path.join(root, name[:-3]))
                            pv = pd.read_csv(os.path.join(root, name))
                            # Save nonzero element indices
                            num_nonzero = 0
                            for i in range(pv.shape[0]):
                                if pv.iloc[i][0] != 0:
                                    num_nonzero += 1
                                    mapped_idx.append(i)
                            self.mapped_indices.append(mapped_idx)

                            # Update last index
                            curr_idx += num_nonzero
                            self.end_indices.append(curr_idx)
                    # Only load first directory (for testing only)
                    if mini and len(self.paths) > 0:
                        print('[WARNING] Only using a small subset of data')
                        break
                    # Save to cache
                    save_obj(cache_file, [self.paths, self.end_indices, self.mapped_indices])
            
            # Compute where validation set starts
            for i in range(len(self.end_indices)):
                # "Round" towards the side of having more validation data
                if self.end_indices[i] > (1-VALIDATION_SPLIT-TEST_SPLIT) * self.end_indices[-1]:
                    self.first_validation_idx = i
                    if i == 0:
                        print('[ERROR] Validation portion is the entire dataset; make sure the dataset is not empty or trivially small.')
                    break

            # Compute where test set starts
            for i in range(self.first_validation_idx+1, len(self.end_indices)):
                # "Round" towards the side of having more test data
                if self.end_indices[i] > (1-TEST_SPLIT) * self.end_indices[-1]:
                    self.first_test_idx = i
                    if i == 0:
                        print('[ERROR] Test portion is the entire dataset; make sure the dataset is not empty or trivially small.')
                    break
        else:
            self.paths = existing_dataset.paths
            self.end_indices = existing_dataset.end_indices
            self.mapped_indices = existing_dataset.mapped_indices
            self.first_validation_idx = existing_dataset.first_validation_idx
            self.first_test_idx = existing_dataset.first_test_idx

    def __len__(self):
        if self.split == 'test':
            return self.end_indices[-1] - self.end_indices[self.first_test_idx - 1]
        if self.split == 'validation':
            return self.end_indices[self.first_test_idx - 1] - self.end_indices[self.first_validation_idx - 1]
        else:
            # train
            return self.end_indices[self.first_validation_idx - 1]
    
    def __getitem__(self, idx):
        # If validation, shift idx over
        if self.split == 'validation':
            idx += self.end_indices[self.first_validation_idx - 1]
        elif self.split == 'test':
            idx += self.end_indices[self.first_test_idx - 1]
            
        # Compute which file this idx corresponds to
        path_idx = bisect.bisect(self.end_indices, idx)
        inner_idx = idx - (self.end_indices[path_idx - 1] if path_idx > 0 else 0)
        # Load audio
        # audio, sr = torchcrepe.load.audio(self.paths[path_idx] + '.wav')
        audio, sr = load_audio(self.paths[path_idx] + '.wav')
        audio = audio.double()
        # from scipy.io import wavfile
        # sr, raw_audio = wavfile.read(self.paths[path_idx] + '.wav')
        raw_audio, sr = torchcrepe.load.audio(self.paths[path_idx] + '.wav') # Uses torchcrepe's audio loading, which uses wavfile rather than librosa
        raw_audio = raw_audio.double()
        # print('audio:', audio)
        # print('raw audio:', raw_audio)
        # print('raw', raw_audio)
        # print('actual', audio)

        # Hop length
        hop_length = int(sr / (1000. / HOP_LENGTH_MS))
        if HOP_LENGTH is not None:
            hop_length = HOP_LENGTH

        # Process audio into format for network
        # generator = torchcrepe.preprocess(audio, sr, hop_length)
        generator = torchcrepe.preprocess(raw_audio, sr, hop_length)
        frame = None
        for frames in generator:
            frame = frames[self.mapped_indices[path_idx][inner_idx]]
            break
            
        # Read pitch vector
        pv = pd.read_csv(self.paths[path_idx] + '.pv')
        pitch = pv.iloc[self.mapped_indices[path_idx][inner_idx]][0]
        pitch = 27.5 * 2 ** ((pitch - 21) / 12) # Convert from MIDI note to frequency
        true_cents = torchcrepe.convert.frequency_to_cents(torch.full((1,),pitch))[0]

        # Convert to bin number and vector of bin probabilities (std dev 25 cents from true frequency)
        label = None
        if not self.strong_penalty:
            label = torch.tensor(np.arange(torchcrepe.PITCH_BINS))
            label = torchcrepe.CENTS_PER_BIN * label + 1997.3794084376191 # Copied from torchcrepe.convert.bins_to_cents, but without dithering
            label = label - true_cents
            label = label * label
            label /= -2 * 25 * 25
            label = np.exp(label)
            label /= sum(label)
        else:
            bin_num = torchcrepe.convert.frequency_to_bins(torch.full((1,), pitch))[0]
            if pitch < MIN_HZ:
                bin_num = 0
            label = torch.zeros((torchcrepe.PITCH_BINS))
            label[bin_num] = 1
        
        return frame, label
    
    # Returns reference pitch in frequency
    def __getpitch__(self, idx):
        # If validation, shift idx over
        if self.split == 'validation':
            idx += self.end_indices[self.first_validation_idx - 1]
        elif self.split == 'test':
            idx += self.end_indices[self.first_test_idx - 1]
            
        # Compute which file this idx corresponds to
        path_idx = bisect.bisect(self.end_indices, idx)
        inner_idx = idx - (self.end_indices[path_idx - 1] if path_idx > 0 else 0)

        # Read pitch vector
        pv = pd.read_csv(self.paths[path_idx] + '.pv')
        pitch = pv.iloc[self.mapped_indices[path_idx][inner_idx]][0]
        pitch = 27.5 * 2 ** ((pitch - 21) / 12) # Convert from MIDI note to frequency
        
        return pitch

if __name__ == '__main__':
    from datetime import datetime
    print('Time before __init__:', datetime.now().strftime('%H:%M:%S'))
    temp = MirDataset(mini=True, strong_penalty=True)
    print('Time before __getitem__:', datetime.now().strftime('%H:%M:%S'))
    print(temp.__getitem__(150))
    print('Time after __getitem__:', datetime.now().strftime('%H:%M:%S'))