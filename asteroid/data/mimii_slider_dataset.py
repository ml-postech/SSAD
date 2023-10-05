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

from .mimii_valve_dataset import MIMIIValveDataset

class MIMIISliderDataset(MIMIIValveDataset):

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
        task_random=False,
        source_random=False,
        num_src_in_mix=2,
    ):

        super().__init__(root, 
            sources=sources,
            targets=targets,
            suffix=suffix,
            split=split,
            subset=subset,
            segment=segment,
            samples_per_track=samples_per_track,
            random_segments=random_segments,
            random_track_mix=random_track_mix,
            source_augmentations=source_augmentations,
            sample_rate=sample_rate,
            normal=normal,
            use_control=use_control,
            task_random=task_random,
            source_random=source_random,
            num_src_in_mix=num_src_in_mix,
            machine_type_dir="slider",
            impulse_label = False,
        )


    def generate_label(self, audio, impulse_label = False):
        # np, [1, 313]
        channels = audio.shape[0]
        rms_fig = librosa.feature.rms(y=audio.numpy())  
        #[c, 1, 313]
        rms_tensor = torch.tensor(rms_fig).permute(0, 2, 1)
        # [channel, time, 1]
        rms_trim = rms_tensor.expand(-1, -1, 512).reshape(channels, -1)[:, :160000]
        # [channel, time]

        min_threshold = (torch.max(rms_trim) + torch.min(rms_trim))/2

        label = (rms_trim > min_threshold).type(torch.float) 
        label = torch.Tensor(scipy.ndimage.binary_dilation(label.numpy(), iterations=3)).type(torch.float) 
        #[channel, time]
        
        if impulse_label:
            time = int(audio.shape[1]//1000) # 0.1 sec
            time = 2
            label_index_lst = torch.nonzero((label[:,1:] - label[:,:-1]) == 1).tolist()
            square_labels = torch.zeros_like(label)
            
            for idx in label_index_lst:
                square_labels[idx[0],(idx[1] + 1):(idx[1] + time)] = 1.0
    
            label = square_labels
        return label
