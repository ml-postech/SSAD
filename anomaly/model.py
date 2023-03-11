import torch
import torch.nn as nn
from asteroid.models import XUMXControl

from utils import bandwidth_to_max_bin

########################################################################
# model
########################################################################

class TorchModel(nn.Module):
    def __init__(self, dim_input):
        super(TorchModel,self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim_input, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim_input),
        )

    def forward(self, x):
        x = self.ff(x)
        return x


class TorchConvModel(nn.Module):
    def __init__(self):
        super(TorchConvModel,self).__init__()
        self.ff = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(4, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=5),
        )

    def forward(self, x):
        assert len(x.shape) == 3
        #[B, T, F]
        x = self.ff(x.unsqueeze(1)).squeeze(1)
        return x

########################################################################

class XUMXSystem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None


def xumx_model(path):
    
    x_unmix = XUMXControl(
        window_length=4096,
        input_mean=None,
        input_scale=None,
        nb_channels=2,
        hidden_size=512,
        in_chan=4096,
        n_hop=1024,
        sources=['s1', 's2'],
        max_bin=bandwidth_to_max_bin(16000, 4096, 16000),
        bidirectional=True,
        sample_rate=16000,
        spec_power=1,
        return_time_signals=True,
    )

    conf = torch.load(path, map_location="cpu")

    system = XUMXSystem()
    system.model = x_unmix

    system.load_state_dict(conf['state_dict'], strict=False)

    return system.model
