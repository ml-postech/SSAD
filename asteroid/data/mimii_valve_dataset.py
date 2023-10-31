from pathlib import Path
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf

from torchaudio import transforms
import librosa
from itertools import product
import numpy as np
import scipy
import pickle
import os

class MIMIIValveDataset(torch.utils.data.Dataset):
    """MUSDB18 music separation dataset

    Folder Structure:
        >>> #0dB/fan/id_00/normal/00000000.wav ---------|
        >>> #0dB/fan/id_02/normal/00000000.wav ---------|
        >>> #0dB/pump/id_00/normal/00000000.wav ---------|

    Args:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names
            that composes the mixture.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        targets (list or None, optional): List of source names to be used as
            targets. If None, a dict with the 4 stems is returned.
             If e.g [`vocals`, `drums`], a tensor with stacked `vocals` and
             `drums` is returned instead of a dict. Defaults to None.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.

    Attributes:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
        tracks (:obj:`list` of :obj:`Dict`): List of track metadata

    References
        "The 2018 Signal Separation Evaluation Campaign" Stoter et al. 2018.
    """

    dataset_name = "MIMII"

    def __init__(
        self,
        root,
        sources=["id_00", "id_02", "id_04", "id_06"],
        targets=None,
        suffix=".wav",
        split="0dB",
        subset=None,
        segment=None,
        samples_per_track=2,
        random_segments=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=16000,
        normal=True,
        use_control=False,
        task_random = False,
        source_random = False,
        num_src_in_mix = 2,
        machine_type_dir = "valve",
        train_ckpt = None,
        val_ckpt = None,
        mode = 'train',
        impulse_label= False
    ):

        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.segment = segment
        self.random_track_mix = random_track_mix
        self.random_segments = random_segments
        self.source_augmentations = source_augmentations
        self.sources = sources
        self.targets = targets
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.normal = normal
        self.source_random = source_random
        self.num_src_in_mix = num_src_in_mix
        self.use_control = use_control 
        self.normal = True
        self.task_random = task_random
        self.machine_type_dir = machine_type_dir
        self.tracks = list(self.get_tracks())
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        self.impulse_label = impulse_label
        
        self.train_ckpt = train_ckpt
        self.val_ckpt = val_ckpt
        self.index_lst = list(range(len(self.tracks)*self.samples_per_track) )
        self.mode = mode
        self.ckpt = self.train_ckpt if self.mode == 'train' else self.val_ckpt
            
        if self.ckpt and os.path.exists(self.ckpt):
            with open(self.ckpt, 'rb') as f:
                self.data = pickle.load(f)
                print("\nCheckpoint loaded successfully.\n")
        else:
            self.data = {index: [] for index in range(len(self.tracks) * self.samples_per_track)}
            print("Checkpoint file not found. Initializing with default data.\n")
            
        self.shift_amount = [0.1, 0.3, 0.5]
        self.overlap_ratio_lst = []
        
    def __getitem__(self, index):
        
        if len(self.data[index]) == 0:
            
            self.index_lst.remove(index)
            
            audio_sources = {}
            active_label_sources = {}
            mix_label = None
            now_overlapped = None

            if self.source_random:
                sources_tmp = ["src1", "src2", "src3", "src4"][:self.num_src_in_mix]
                target_tmp = sources_tmp
            else:
                sources_tmp = self.sources
                target_tmp = self.targets

            # get track_id
            track_id = index // self.samples_per_track
            if self.random_segments:
                start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
            else:
                start = 0

            # load sources
            for i, source in enumerate(sources_tmp):
                if self.source_random:
                    src_idx = 0
                else:
                    src_idx = i

                # optionally select a random track for each source
                if self.random_track_mix or self.source_random:
                    # load a different track
                    track_id = random.choice(range(len(self.tracks)))
                    if self.random_segments:
                        start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)

                # loads the full track duration
                start_sample = int(start * self.sample_rate)
                # check if dur is none
                if self.segment:
                    # stop in soundfile is calc in samples, not seconds
                    stop_sample = start_sample + int(self.segment * self.sample_rate)
                else:
                    # set to None for reading complete file
                    stop_sample = None

                # load actual audio
                np_audio, sr = sf.read(
                    Path(self.tracks[track_id]["source_paths"][src_idx]),
                    always_2d=True,
                    start=start_sample,
                    stop=stop_sample,
                )
                # convert to torch tensor
                audio = torch.tensor(np_audio.T, dtype=torch.float)[:, :]
                
                # apply source-wise augmentations
                audio = self.source_augmentations(audio)
                
                #[channel, time]
                
                label = self.generate_label(audio, impulse_label = self.impulse_label)
                #[channel, time]
                
                # calculate overlap ratio and shift
                # mix_label = 현재 mixture에서 active한 부분 
                if mix_label == None:
                    mix_label = label
                    now_overlapped = label
                    
                else:
                    y_candidates = [self.shift(audio, -self.shift_amount[2] * sr), 
                                self.shift(audio, -self.shift_amount[1] * sr), 
                                self.shift(audio, -self.shift_amount[0] * sr), 
                                audio, 
                                self.shift(audio, self.shift_amount[0] * sr),
                                self.shift(audio, self.shift_amount[1] * sr),
                                self.shift(audio, self.shift_amount[2] * sr),
                                ]
                    
                    label_candidates = list(map(lambda x: self.generate_label(x), y_candidates))
                    overlap_candidates = list(map(lambda x: np.logical_and(x, mix_label).sum() / np.logical_or(x, mix_label).sum(), label_candidates))
                    
                    target_idx = np.argmax(overlap_candidates)
                    label = label_candidates[target_idx]
                    mix_label = np.logical_or(mix_label, label)
                    now_overlapped = now_overlapped + label
                    audio = y_candidates[target_idx]

                if self.use_control:
                    audio_sources[source] = audio
                    active_label_sources[source] = label
                else:
                    audio_sources[source] = audio
                

            # apply linear mix over source index=0
            # make mixture for i-th channel and use 0-th chnnel as gt
            if self.task_random:
                targets = target_tmp.copy()
                random.shuffle(targets)       
                if len(targets) > self.num_src_in_mix:
                    targets = targets[:self.num_src_in_mix]
            else:
                targets = target_tmp
            audioes = torch.stack([audio_sources[src] for src in targets])
            audio_mix = torch.stack([audioes[i, 0:2, :] for i in range(len(targets))]).sum(0)
            
            if targets:
                audio_sources = audioes[:, 0:2, :]

            if self.use_control:
                active_labels = torch.stack([active_label_sources[src] for src in targets])
                # [source, channel, time]
                if targets:
                    active_labels = active_labels[:, 0:2, :]
                
                self.data[index] = torch.zeros(1 + 2*self.num_src_in_mix, audio_sources.shape[1], audio_sources.shape[2])
                self.data[index][0:1,:, :] = audio_mix.unsqueeze(0)
                self.data[index][1:self.num_src_in_mix+1,:, :] = audio_sources
                self.data[index][self.num_src_in_mix+1:2*self.num_src_in_mix+1,:, :] = active_labels
                now_overlapped = torch.where(now_overlapped>1, torch.tensor(1), torch.tensor(0))
                overlap_ratio = now_overlapped.sum()/mix_label.sum()
                self.overlap_ratio_lst.append(overlap_ratio)
                
                if len(self.index_lst) == 0:
                    self.save_dictionary("{machine_type}_{mode}_control_overlap.pkl".format(machine_type = self.machine_type_dir, mode = self.mode))
                
                return audio_mix, audio_sources, active_labels
            
            self.data[index] = torch.zeros(1 + self.num_src_in_mix, audio_sources.shape[1], audio_sources.shape[2])
            self.data[index][0:1,:, :] = audio_mix.unsqueeze(0)
            self.data[index][1:self.num_src_in_mix+1,:, :] = audio_sources            
            return audio_mix, audio_sources
        
        else:
            if self.use_control:
                audio_mix = self.data[index][0:1,:, :]
                audio_mix = audio_mix.squeeze(0)
                audio_sources = self.data[index][1:self.num_src_in_mix+1,:, :]
                active_labels = self.data[index][self.num_src_in_mix+1:2*self.num_src_in_mix+1,:, :]
    
                return audio_mix, audio_sources, active_labels
            else:
                audio_mix = self.data[index][0:1,:, :]
                audio_mix = audio_mix.squeeze(0)
                audio_sources = self.data[index][1:self.num_src_in_mix+1,:, :]         
            return audio_mix, audio_sources
        
                    
            
        

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        pp = []
        if self.source_random:
            for src in self.sources:
                pp.extend(p.glob(f'{self.machine_type_dir}/{src}/{"normal" if self.normal else "abnormal"}/*.wav'))
        else:
            pp.extend(p.glob(f'{self.machine_type_dir}/id_00/{"normal" if self.normal else "abnormal"}/*.wav'))

        for track_path in tqdm.tqdm(pp):
            if self.subset and track_path.stem not in self.subset:
                # skip this track
                continue
            
            if self.source_random:
                source_paths = [track_path]
            else:
                source_paths = [Path(str(track_path).replace(self.sources[0], s)) for s in self.sources]
               
            if not all(sp.exists() for sp in source_paths):
                print("Exclude track due to non-existing source", track_path)
                continue

            # get metadata
            infos = list(map(sf.info, source_paths))
            if not all(i.samplerate == self.sample_rate for i in infos):
                print("Exclude track due to different sample rate ", track_path)
                continue

            if self.segment is not None:
                # get minimum duration of track
                min_duration = min(i.duration for i in infos)
                if min_duration > self.segment:
                    yield ({"path": track_path, "min_duration": min_duration, "source_paths": source_paths})
            else:
                yield ({"path": track_path, "min_duration": None, "source_paths": source_paths})

              
    def generate_label(self, audio, impulse_label = False):
        # np, [c, t]
        time = audio.shape[-1]
        channels = audio.shape[0]
        rms_fig = librosa.feature.rms(y=audio.numpy()) 
        #[c, 1, 313]

        rms_tensor = torch.tensor(rms_fig).permute(0, 2, 1)
        # [channel, time, 1]
        rms_trim = rms_tensor.expand(-1, -1, 512).reshape(channels, -1)[:, :time]
        # [channel, time]

        k = int(audio.shape[1]*0.8)
        min_threshold, _ = torch.kthvalue(rms_trim[0, :], k)

        label = (rms_trim > min_threshold).type(torch.float) 
        # label = torch.Tensor(scipy.ndimage.binary_dilation(label.numpy(), iterations=3)).type(torch.float) 
        #[channel, time]
        
        ##############################################################################
        if impulse_label:
            time = int(audio.shape[1]//100) # 0.1 sec
            time = 2
            label_index_lst = torch.nonzero((label[:,1:] - label[:,:-1]) == 1).tolist()
            square_labels = torch.zeros_like(label)
            
            for idx in label_index_lst:
                square_labels[idx[0],(idx[1] + 1):(idx[1] + time)] = 1.0
            label = square_labels
        ##############################################################################
        return label
    
    
    def shift(self, data, amount_to_right):
        amount_to_right = int(amount_to_right)
        if amount_to_right > 0:
            new_data = np.concatenate([np.zeros_like(data[:, :amount_to_right]), data[:, :-amount_to_right]], axis = 1)
        elif amount_to_right < 0:
            new_data = np.concatenate([data[:, amount_to_right:], np.zeros_like(data[:, :amount_to_right])], axis = 1)
        else:
            new_data = data
        
        return torch.tensor(new_data)
    
    def save_dictionary(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
            

            
    def shift(self, data, amount_to_right):
        amount_to_right = int(amount_to_right)
        white_noise = 0.001* torch.randn(data.shape)
        if amount_to_right > 0:
            new_data = np.concatenate([white_noise[:, :amount_to_right], data[:, :-amount_to_right]], axis = 1)
        elif amount_to_right < 0:
            new_data = np.concatenate([data[:, amount_to_right:], white_noise[:, :amount_to_right]], axis = 1)
        else:
            new_data = data
        
        return torch.tensor(new_data)
    
 