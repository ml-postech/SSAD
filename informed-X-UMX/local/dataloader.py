from asteroid.data import MIMIISliderDataset, MIMIIValveDataset
import torch
from pathlib import Path
import numpy as np
import random

train_tracks = [f"{n:0>3}" for n in range(10, 350)]




def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()

    dataset_kwargs = {
        "root": Path(args.train_dir),
    }

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    if args.machine_type == 'valve':
        Dataset = MIMIIValveDataset
        validation_tracks = validation_tracks = ["00000000", "00000001","00000002", "00000003"]
    elif args.machine_type == 'slider':
        Dataset = MIMIISliderDataset
        validation_tracks = validation_tracks = ["00000000", "00000001","00000002", "00000003"]
    else:
        raise Exception("unexpected machine type")
    
    
    train_dataset = Dataset(
        split=args.split,
        sources=args.sources,
        targets=args.sources,
        source_augmentations=source_augmentations,
        random_track_mix=True,
        segment=args.seq_dur,
        random_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        use_control=args.use_control,
        task_random=args.task_random,
        source_random=args.source_random,
        num_src_in_mix=args.num_src_in_mix,
        train_ckpt = args.train_ckpt,
        val_ckpt = args.val_ckpt,
        mode = 'train',
        **dataset_kwargs,
    )
    
    
    train_dataset = filtering_out_valid(train_dataset, validation_tracks)
    
    valid_dataset = Dataset(
        split=args.split,
        subset=validation_tracks,
        sources=args.sources,
        targets=args.sources,
        source_augmentations=source_augmentations,
        segment=args.val_dur,
        sample_rate=args.sample_rate,
        use_control=args.use_control,
        task_random=args.task_random,
        source_random=args.source_random,
        num_src_in_mix=args.num_src_in_mix,
        train_ckpt = args.train_ckpt,
        val_ckpt = args.val_ckpt,
        mode = 'val',
        **dataset_kwargs,

    )

    return train_dataset, valid_dataset


def filtering_out_valid(input_dataset, validation_tracks):
    """Filtering out validation tracks from input dataset.

    Return:
        input_dataset (w/o validation tracks)
    """
    input_dataset.tracks = [
        tmp
        for tmp in input_dataset.tracks
        if not (str(tmp["path"]).split("/")[-1] in validation_tracks)
    ]
    return input_dataset


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


def _augment_delay(audio, max=16000):
    """Applies a random gain to each source between `low` and `high`"""
    delay = random.randint(0, max)
    audio_len = audio.shape[1]
    delayed = torch.cat([torch.zeros_like(audio)[:, :delay], audio[:, :audio_len - delay]], dim=1)
    return delayed


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`"""
    gain = low + torch.rand(1) * (high - low)
    return audio * gain


def _augment_channelswap(audio):
    """Randomly swap channels of stereo sources"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])

    return audio
