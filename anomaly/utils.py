import pickle
import os
import sys
import glob

import numpy
import numpy as np
import librosa
import librosa.core
import librosa.feature
import scipy
import yaml
import logging
from tqdm import tqdm
import random
import torch


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################


########################################################################
# file I/O
########################################################################
# pickle I/O
def save_pickle(filename, save_data):
    """
    picklenize the data.
    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized
    return : None
    """
    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    """
    unpicklenize the data.
    filename : str
        pickle filename
    return : data
    """
    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data


# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.
    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data
    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=1):
    """
    demux .wav file.
    wav_name : str
        target .wav file
    channel : int
        target channel number
    return : numpy.array( float )
        demuxed mono data
    Enabled to read multiple sampling rates.
    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        if channel > 1:
            return sr, numpy.array(multi_channel_data)[:channel, :]
        else:
            return sr, numpy.array(multi_channel_data)[0, :]

    except ValueError as msg:
        logger.warning(f'{msg}')

def file_to_wav_stereo(file_name):
    sr, y = demux_wav(file_name, channel=2)
    return sr, y

########################################################################

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.
        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.
        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.
        name : str
            save .png file path.
        return : None
        """
        self.plt.savefig(name)


########################################################################


def fix_seed(seed: int = 42):
    random.seed(seed) # random
    numpy.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


########################################################################
# feature extractor
########################################################################

def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = numpy.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)

    return numpy.max(numpy.where(freqs <= bandwidth)[0]) + 1


def get_overlap_ratio(signal1, signal2):
    return torch.sum(torch.logical_and(signal1, signal2)) / torch.sum(torch.logical_or(signal1, signal2))


def generate_label(y, machine='valve'):
    # np, [c, t]
    channels = y.shape[0]
    frames = 5
    rms_fig = librosa.feature.rms(y=y) 
    #[c, 1, 313]

    rms_tensor = torch.tensor(rms_fig).permute(0, 2, 1)
    # [channel, time, 1]
    rms_trim = rms_tensor.expand(-1, -1, 512).reshape(channels, -1)[:, :160000]
    # [channel, time]

    rms_trim_spec = torch.stack([torch.tensor(rms_fig[:, 0, i:i+rms_fig.shape[2]-frames+1]) for i in range(frames)], dim=2)
    #[c, 313-4, 5]

    if machine == 'valve':
        k = int(y.shape[1]*0.8)
        min_threshold, _ = torch.kthvalue(rms_trim[0, :], k)
    else:
        min_threshold = (torch.max(rms_trim) + torch.min(rms_trim))/2

    label = (rms_trim > min_threshold).type(torch.float) 
    #[channel, time]
    label_spec = (rms_trim_spec > min_threshold).type(torch.float) 
    if machine == 'slider':
        label_spec = torch.Tensor(scipy.ndimage.binary_dilation(label_spec.numpy(), iterations=3)).type(torch.float) 
    return label, label_spec


def wav_to_spec_vector_array(sr, y,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         spec_mask=None):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = numpy.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    if spec_mask is not None:
        assert spec_mask.shape[0] == vectorarray.shape[0]
        assert spec_mask.shape[1] == vectorarray.shape[1]
        vectorarray = numpy.multiply(vectorarray, spec_mask)

    return vectorarray


def file_to_spec_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):

    sr, y = demux_wav(file_name)
    return wav_to_spec_vector_array(sr, y, 
                         n_mels,
                         frames,
                         n_fft,
                         hop_length,
                         power)

def file_to_masked_spec_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         machine='valve'):

    sr, y = demux_wav(file_name)
    _, spec_label = generate_label(np.expand_dims(y, axis=0), machine)
    spec_label = spec_label.unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0).numpy()
    # [309, 320]
    
    vector_array = wav_to_spec_vector_array(sr, y, n_mels, frames, n_fft, hop_length, power, spec_mask=spec_label)
    
    return vector_array, spec_label


def wav_to_spec_vector_2d_array(sr, y,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, frames, n_mels), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = numpy.zeros((vectorarray_size, frames, n_mels), float)
    for t in range(frames):
        vectorarray[:, t, :] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray

def file_to_spec_vector_2d_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):

    sr, y = demux_wav(file_name)
    return wav_to_spec_vector_2d_array(sr, y, 
                         n_mels,
                         frames,
                         n_fft,
                         hop_length,
                         power)
