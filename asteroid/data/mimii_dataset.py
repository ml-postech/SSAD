from pathlib import Path
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf

from torchaudio import transforms
import librosa

class MIMIIDataset(torch.utils.data.Dataset):
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
        sources=["fan", "pump", "slider", "valve"],
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
        self.tracks = list(self.get_tracks())
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        self.use_control = use_control
        self.normal = True 
        
    def __getitem__(self, index):
       
        audio_sources = {}
        active_label_sources = {}

        # get track_id
        track_id = index // self.samples_per_track
        if self.random_segments:
            start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
        else:
            start = 0

        # load sources
        for i, source in enumerate(self.sources):
            # optionally select a random track for each source
            if self.random_track_mix:
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
            audio, _ = sf.read(
                Path(self.tracks[track_id]["source_paths"][i]),
                always_2d=True,
                start=start_sample,
                stop=stop_sample,
            )
            # convert to torch tensor
            audio = torch.tensor(audio.T, dtype=torch.float)[:, :]
            # apply source-wise augmentations
            audio = self.source_augmentations(audio)

            #apply mask
            audio_len = audio.shape[1]
            mask_len = random.randrange(audio_len//2)
            start_point = random.randrange(0, audio_len - mask_len)
            torch.clamp_(audio[:, start_point:start_point + mask_len], min=-0.001, max=0.001)
            audio_sources[source] = audio
            
            if self.use_control:
                active_label = torch.ones_like(audio)
                active_label[:, start_point:start_point + mask_len] = 0
                active_label_sources[source] = active_label
                # [channel, time]

                # #make mean label
                # dur_chunk = audio_len//1000
                # audio_threashold = 0.01
                # active_label = torch.empty((2, dur_chunk))
                # num_chunk = audio_len//dur_chunk
                # if audio_len % dur_chunk != 0:
                #     num_chunk = num_chunk + 1
                
                # for i in range(num_chunk):
                #     chunk = audio[:, i*dur_chunk:(i+1)*dur_chunk]
                #     chunk_label = torch.empty_like(chunk[:2, :])
                #     for j in range(2):
                #         if torch.mean(torch.abs(chunk[j, :])) < audio_threashold:
                #             chunk_label[j, :] = torch.zeros_like(chunk[j, :])              
                #         else:
                #             chunk_label[j, :] = torch.ones_like(chunk[j, :])
                #     active_label = torch.cat((active_label, chunk_label), dim = 1)
                # active_label_sources[source] = active_label[:, dur_chunk:]

            

        # apply linear mix over source index=0
        # make mixture for i-th channel and use 0-th chnnel as gt
        audioes = torch.stack([audio_sources[src] for src in self.targets])
        audio_mix = torch.stack([audioes[i, 2 * i : 2 * i + 2, :] for i in range(len(self.sources))]).sum(0)
        
        if self.targets:
            audio_sources = audioes[:, 0:2, :]
            for i in range(len(self.sources)):
                audio_sources[i, :, :] = audioes[i, 2 * i : 2 * i + 2, :]

        if self.use_control:
            active_labels = torch.stack([active_label_sources[src] for src in self.targets])
            # [source, channel, time]
            if self.targets:
                active_labels = active_labels[:, 0:2, :]

            return audio_mix, audio_sources, active_labels


        return audio_mix, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        ids = ["id_00", "id_02", "id_04"]
        p = Path(self.root, self.split)
        pp = []
        for id in ids:
            pp.extend(p.glob(f'fan/{id}/{"normal" if self.normal else "abnormal"}/*.wav'))
        
        for track_path in tqdm.tqdm(pp):
            # print(track_path)
            if self.subset and track_path.stem not in self.subset:
                # skip this track
                continue
            
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
