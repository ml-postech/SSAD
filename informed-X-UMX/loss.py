import itertools
from itertools import permutations
from operator import itemgetter

import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from torch import nn

from asteroid.models.x_umx import _STFT, _Spectrogram
from asteroid.losses.pit_wrapper import PITLossWrapper
from asteroid.losses import singlesrc_mse


class CustomPITLossWrapper(PITLossWrapper):
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None):
        super().__init__(loss_func, pit_from=pit_from, perm_reduce=perm_reduce)
        if self.pit_from not in ["perm_avg"]:
            raise ValueError(
                "Unsupported loss function type for now. Expected"
                "one of [`perm_avg`]."
                "CustomPITLossWrapper needs to be fixed."
            )
    
    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        r"""Find the best permutation and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape $(batch, nsrc, ...)$.
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            reduce_kwargs (dict or None): kwargs that will be passed to the
                pairwise losses reduce function (`perm_reduce`).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
              the batch.
            - The reordered targets estimates if ``return_est`` is True.
              :class:`torch.Tensor` of shape $(batch, nsrc, ...)$.
        """
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == "perm_avg":
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )

            reordered = self.reorder_source(est_targets, batch_indices)
            loss = self.loss_func(reordered, targets)
            mean_loss = torch.mean(loss)
            if return_est:
                return mean_loss, reordered
            else:
                return mean_loss
        else:
            print("Not implemented for CustomPITWrapper")
            assert False

    @staticmethod
    def best_perm_from_perm_avg_loss(loss_func, est_targets, targets, **kwargs):
        r"""Find best permutation from loss function with source axis.

        Args:
            loss_func: function with signature $(est_targets, targets, **kwargs)$
                The loss function batch losses from.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
                The loss corresponding to the best permutation of size $(batch,)$.

            - :class:`torch.Tensor`:
                The indices of the best permutations.
        """

        
        """est_targets (list) has 2 elements:
            [0]->Estimated Spec. : (Sources, Frames, Batch size, Channels, Freq. bins)
            [1]->Estimated Signal: (Sources, Batch size, Channels, Time Length)

        targets: (Batch, Source, Channels, TimeLen)
        """
        n_src = targets.shape[1]
        assert est_targets[0].shape[0] == n_src
        perms = torch.tensor(list(permutations(range(n_src))), dtype=torch.long)
        loss_set = torch.stack(
            [loss_func((est_targets[0][perm], est_targets[1][perm]), targets, **kwargs) for perm in perms], 
            dim=1,
        )
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)
        # Permutation indices for each batch.
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices
        
    @staticmethod
    def reorder_source(source, batch_indices):
        r"""Reorder sources according to the best permutation.

        Args:
            source (torch.Tensor): Tensor of shape :math:`(batch, n_src, time)`
            batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
                Contains optimal permutation indices for each batch.

        Returns:
            :class:`torch.Tensor`: Reordered sources.
        """
        """est_targets (list) has 2 elements:
            [0]->Estimated Spec. : (Sources, Frames, Batch size, Channels, Freq. bins)
            [1]->Estimated Signal: (Sources, Batch size, Channels, Time Length)
        """
        reordered_sources_f = torch.stack(
            [source[0][src_idx, :, b, :, :] for b, src_idx in enumerate(batch_indices)],
            dim=2
        )
        reordered_sources_t = torch.stack(
            [source[1][src_idx, b, :, :] for b, src_idx in enumerate(batch_indices)],
            dim=1
        )
        return (reordered_sources_f, reordered_sources_t)


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def freq_domain_loss(s_hat, gt_spec, combination=True):
    """Calculate frequency-domain loss between estimated and reference spectrograms.
    MSE between target and estimated target spectrograms is adopted as frequency-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Input:
        estimated spectrograms
            (Sources, Freq. bins, Batch size, Channels, Frames)
        reference spectrograms
            (Freq. bins, Batch size, Sources x Channels, Frames)
        whether use combination or not (optional)
    Output:
        calculated frequency-domain loss
    """

    n_src = len(s_hat)
    idx_list = [i for i in range(n_src)]

    inferences = []
    refrences = []
    for i, s in enumerate(s_hat):
        inferences.append(s)
        refrences.append(gt_spec[..., 2 * i : 2 * i + 2, :])
    assert inferences[0].shape == refrences[0].shape

    _loss_mse = 0.0
    cnt = 0.0
    for i in range(n_src):
        _loss_mse += singlesrc_mse(inferences[i].transpose(0,1), refrences[i].transpose(0,1))
        cnt += 1.0

    # If Combination is True, calculate the expected combinations.
    if combination:
        for c in range(2, n_src):
            patterns = list(itertools.combinations(idx_list, c))
            for indices in patterns:
                tmp_loss = singlesrc_mse(
                    sum(itemgetter(*indices)(inferences)).transpose(0,1),
                    sum(itemgetter(*indices)(refrences)).transpose(0,1),
                )
                _loss_mse += tmp_loss
                cnt += 1.0

    _loss_mse /= cnt

    return _loss_mse


def time_domain_loss(mix, time_hat, gt_time, combination=True):
    """Calculate weighted time-domain loss between estimated and reference time signals.
    weighted SDR [1] between target and estimated target signals is adopted as time-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Input:
        mixture time signal
            (Batch size, Channels, Time Length (samples))
        estimated time signals
            (Sources, Batch size, Channels, Time Length (samples))
        reference time signals
            (Batch size, Sources x Channels, Time Length (samples))
        whether use combination or not (optional)
    Output:
        calculated time-domain loss

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
          Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    n_src, n_batch, n_channel, time_length = time_hat.shape
    idx_list = [i for i in range(n_src)]

    # Fix Length
    mix = mix[Ellipsis, :time_length]
    gt_time = gt_time[Ellipsis, :time_length]

    # Prepare Data and Fix Shape
    mix_ref = [mix]
    mix_ref.extend([gt_time[..., 2 * i : 2 * i + 2, :] for i in range(n_src)])
    mix_ref = torch.stack(mix_ref)
    mix_ref = mix_ref.view(-1, time_length)
    time_hat = time_hat.view(n_batch * n_channel * time_hat.shape[0], time_hat.shape[-1])

    # If Combination is True, calculate the expected combinations.
    if combination:
        indices = []
        for c in range(2, n_src):
            indices.extend(list(itertools.combinations(idx_list, c)))

        for tr in indices:
            sp = [n_batch * n_channel * (tr[i] + 1) for i in range(len(tr))]
            ep = [n_batch * n_channel * (tr[i] + 2) for i in range(len(tr))]
            spi = [n_batch * n_channel * tr[i] for i in range(len(tr))]
            epi = [n_batch * n_channel * (tr[i] + 1) for i in range(len(tr))]

            tmp = sum([mix_ref[sp[i] : ep[i], ...].clone() for i in range(len(tr))])
            tmpi = sum([time_hat[spi[i] : epi[i], ...].clone() for i in range(len(tr))])
            mix_ref = torch.cat([mix_ref, tmp], dim=0)
            time_hat = torch.cat([time_hat, tmpi], dim=0)

        mix_t = mix_ref[: n_batch * n_channel, Ellipsis].repeat(n_src + len(indices), 1)
        refrences_t = mix_ref[n_batch * n_channel :, Ellipsis]
    else:
        mix_t = mix_ref[: n_batch * n_channel, Ellipsis].repeat(n_src, 1)
        refrences_t = mix_ref[n_batch * n_channel :, Ellipsis]

    # Calculation
    _loss_sdr = weighted_sdr(time_hat, refrences_t, mix_t)

    return 1.0 + _loss_sdr


def weighted_sdr(input, gt, mix, weighted=True, eps=1e-10):
    # ``input'', ``gt'' and ``mix'' should be (Batch, Time Length)
    assert input.shape == gt.shape
    assert mix.shape == gt.shape

    ns = mix - gt
    ns_hat = mix - input

    if weighted:
        alpha_num = (gt * gt).sum(1, keepdims=True)
        alpha_denom = (gt * gt).sum(1, keepdims=True) + (ns * ns).sum(1, keepdims=True)
        alpha = alpha_num / (alpha_denom + eps)
    else:
        alpha = 0.5

    # Target
    num_cln = (input * gt).sum(1, keepdims=True)
    denom_cln = torch.sqrt(eps + (input * input).sum(1, keepdims=True)) * torch.sqrt(
        eps + (gt * gt).sum(1, keepdims=True)
    )
    sdr_cln = num_cln / (denom_cln + eps)

    # Noise
    num_noise = (ns * ns_hat).sum(1, keepdims=True)
    denom_noise = torch.sqrt(eps + (ns_hat * ns_hat).sum(1, keepdims=True)) * torch.sqrt(
        eps + (ns * ns).sum(1, keepdims=True)
    )
    sdr_noise = num_noise / (denom_noise + eps)

    return torch.mean(-alpha * sdr_cln - (1.0 - alpha) * sdr_noise, dim=0)


class MultiDomainLoss(_Loss):
    """A class for calculating loss functions of X-UMX.

    Args:
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        spec_power (int): Exponent for spectrogram calculation.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        loss_combine_sources (bool): Set to true if you are using the combination scheme
            proposed in [1]. If you select ``loss_combine_sources: yes'' via
            conf.yml, this is set as True.
        loss_use_multidomain (bool): Set to true if you are using a frequency- and time-domain
            losses collaboratively, i.e., Multi Domain Loss (MDL) proposed in [1].
            If you select ``loss_use_multidomain: yes'' via conf.yml, this is set as True.
        mix_coef (float): A mixing parameter for multi domain losses

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(
        self,
        window_length,
        in_chan,
        n_hop,
        spec_power,
        nb_channels,
        loss_combine_sources,
        loss_use_multidomain,
        mix_coef,
        reduce='mean'
    ):
        super().__init__()
        self.transform = nn.Sequential(
            _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop),
            _Spectrogram(spec_power=spec_power, mono=(nb_channels == 1)),
        )
        self._combi = loss_combine_sources
        self._multi = loss_use_multidomain
        self.coef = mix_coef
        print("Combination Loss: {}".format(self._combi))
        if self._multi:
            print(
                "Multi Domain Loss: {}, scaling parameter for time-domain loss={}".format(
                    self._multi, self.coef
                )
            )
        else:
            print("Multi Domain Loss: {}".format(self._multi))
        self.cnt = 0
        self.reduce = reduce

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """est_targets (list) has 2 elements:
            [0]->Estimated Spec. : (Sources, Frames, Batch size, Channels, Freq. bins)
            [1]->Estimated Signal: (Sources, Batch size, Channels, Time Length)

        targets: (Batch, Source, Channels, TimeLen)
        """

        spec_hat = est_targets[0]
        time_hat = est_targets[1]

        # Fix shape and apply transformation of targets
        n_batch, n_src, n_channel, time_length = targets.shape
        targets = targets.view(n_batch, n_src * n_channel, time_length)
        Y = self.transform(targets)[0]

        if self._multi:
            n_src = spec_hat.shape[0]
            mixture_t = sum([targets[:, 2 * i : 2 * i + 2, ...] for i in range(n_src)])
            loss_f = freq_domain_loss(spec_hat, Y, combination=self._combi)
            loss_t = time_domain_loss(mixture_t, time_hat, targets, combination=self._combi)
            loss = float(self.coef) * loss_t + loss_f
        else:
            loss = freq_domain_loss(spec_hat, Y, combination=self._combi)

        if self.reduce == 'mean':
            loss = loss.mean()

        if return_est:
            return loss, est_targets
        else:
            return loss
