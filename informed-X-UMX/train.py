import os
import argparse
import json
import random
import copy
import tqdm
import numpy as np
import sklearn.preprocessing
import museval

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.engine.system import System
from asteroid.engine.optimizers import make_optimizer
from asteroid.models import XUMXControl, XUMX
from asteroid.models.x_umx import _STFT, _Spectrogram

from local import dataloader
from pathlib import Path
from loss import MultiDomainLoss, CustomPITLossWrapper

from pytorch_lightning.loggers import WandbLogger
import wandb 

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]


parser = argparse.ArgumentParser()
parser.add_argument("--conf", action="store", default="local/conf.yml")
def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def get_statistics(args, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        _STFT(window_length=args.window_length, n_fft=args.in_chan, n_hop=args.nhop),
        _Spectrogram(spec_power=args.spec_power, mono=True),
    )

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.random_segments = False
    dataset_scaler.random_track_mix = False
    dataset_scaler.segment = False
    pbar = tqdm.tqdm(range(len(dataset_scaler)))
    for ind in pbar:
        if args.use_control:
            x, _, _ = dataset_scaler[ind]
        else:
            x, _= dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])[0]
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


class XUMXManager(System):
    """A class for X-UMX systems inheriting the base system class of lightning.
    The difference from base class is specialized for X-UMX to calculate
    validation loss preventing the GPU memory over flow.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
        val_dur (float): When calculating validation loss, the loss is calculated
            per this ``val_dur'' in seconds on GPU to prevent memory overflow.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
        val_dur=None,
    ):
        config["data"].pop("sources")
        config["data"].pop("source_augmentations")
        super().__init__(model, optimizer, loss_func, train_loader, val_loader, scheduler, config)
        self.val_dur_samples = model.sample_rate * val_dur

    def common_step(self, batch, batch_nb, train=True, return_est=False):
        inputs, targets = batch
        est_targets = self(inputs)
        if return_est:
            loss, perm_est_targets = self.loss_func(est_targets, targets, return_est=return_est)
            return loss, perm_est_targets
        else:
            loss = self.loss_func(est_targets, targets)
            return loss

    def validation_step(self, batch, batch_nb):
        """
        We calculate the ``validation loss'' by splitting each song into
        smaller chunks in order to prevent GPU out-of-memory errors.
        The length of each chunk is given by ``self.val_dur_samples'' which is
        computed from ``sample_rate [Hz]'' and ``val_dur [seconds]'' which are
        both set in conf.yml.
        """
        sp = 0
        dur_samples = int(self.val_dur_samples)
        cnt = 0
        loss_tmp = 0.0
        sdr_tmp = 0.0
        sdri_tmp = 0.0

        while 1:
            input = batch[0][Ellipsis, sp : sp + dur_samples]
            gt = batch[1][Ellipsis, sp : sp + dur_samples]
            batch_tmp = [
                input,  # input
                gt,  # target
            ]
            loss_step, est_step = self.common_step(batch_tmp, batch_nb, train=False, return_est=True)
            loss_tmp += loss_step
            cnt += 1
            sp += dur_samples

            ###
            if self.current_epoch % 10 == 0:
                targets = gt
                mixture = input
                spec_hat, time_hat = est_step

                
                n_src, n_batch, n_channel, time_length = time_hat.shape
                assert n_batch == 1
                time_hat = time_hat.squeeze(1).permute(0, 2, 1)

                targets = targets[:, :, :, :time_length]
                targets = targets.squeeze(0).permute(0, 2, 1)

                mixture = mixture[:, :, :time_length]
                mixture = mixture.permute(0, 2, 1)
                mix_audio = mixture.clone()
                mixture = mixture.repeat(n_src, 1, 1)

                # sdr_mix, _, _, _ = museval.evaluate(targets.detach().cpu(), mixture.detach().cpu())
                sdr, _, _, _ = museval.evaluate(targets.detach().cpu(), time_hat.detach().cpu())

                sdr_tmp += np.mean(sdr, axis=1)
                # sdri_tmp += np.mean(sdr - sdr_mix, axis=1)

            if batch_tmp[0].shape[-1] < dur_samples or batch[0].shape[-1] == cnt * dur_samples:
                break
        loss = loss_tmp / cnt
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        if self.current_epoch % 10 == 0:
            sdr = sdr_tmp / cnt
            # sdri = sdri_tmp / cnt
            for i, src in enumerate(self.model.sources): 
                self.log(f"val_SDR_{src}", sdr[i], on_epoch=True, prog_bar=True)
                # self.log(f"val_SDRi_{src}", sdri[i], on_epoch=True, prog_bar=True)

            if batch_nb % 8 == 0:
                valve1_hat = np.array(time_hat[0, :, 0].reshape(-1,1).detach().cpu())
                valve2_hat = np.array(time_hat[1, :, 0].reshape(-1,1).detach().cpu())

                valve1_gt = np.array(targets[0, :, 0].reshape(-1,1).detach().cpu())
                valve2_gt = np.array(targets[1, :, 0].reshape(-1,1).detach().cpu())

                mixture_audio = np.array(mix_audio[0, :, 0].reshape(-1,1).detach().cpu())

                self.logger.experiment.log({
                    "val_valve1": [wandb.Audio(valve1_hat, sample_rate = 16000)],
                    "gt_valve1": [wandb.Audio(valve1_gt, sample_rate = 16000)],
                    "val_valve2": [wandb.Audio(valve2_hat, sample_rate = 16000)],
                    "gt_valve2": [wandb.Audio(valve2_gt, sample_rate = 16000)],
                    "mixture": [wandb.Audio(mixture_audio, sample_rate = 16000)],
                })
                
            self.log("val_mean_SDR", np.mean(sdr), on_epoch=True, prog_bar=True)
            # self.log("val_mean_SDRi", np.mean(sdri), on_epoch=True, prog_bar=True)


class XUMXControlManager(XUMXManager):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
        val_dur=None,
    ):
        super().__init__(model, optimizer, loss_func, train_loader, val_loader, scheduler, config, val_dur)

    def common_step(self, batch, batch_nb, train=True, return_est=False):
        inputs, targets, labels = batch
        est_targets = self(inputs, labels)
        if return_est:
            loss, perm_est_targets = self.loss_func(est_targets, targets, return_est=return_est)
            return loss, perm_est_targets
        else:
            loss = self.loss_func(est_targets, targets)
            return loss

    def validation_step(self, batch, batch_nb):
        """
        We calculate the ``validation loss'' by splitting each song into
        smaller chunks in order to prevent GPU out-of-memory errors.
        The length of each chunk is given by ``self.val_dur_samples'' which is
        computed from ``sample_rate [Hz]'' and ``val_dur [seconds]'' which are
        both set in conf.yml.
        """
        sp = 0
        dur_samples = int(self.val_dur_samples)
        cnt = 0
        loss_tmp = 0.0
        sdr_tmp = 0.0
        sdri_tmp = 0.0

        while 1:
            input = batch[0][Ellipsis, sp : sp + dur_samples]
            gt = batch[1][Ellipsis, sp : sp + dur_samples]
            label = batch[2][Ellipsis, sp : sp + dur_samples]
            batch_tmp = [
                input,  # input
                gt,  # target
                label,
            ]

            loss_step, est_step = self.common_step(batch_tmp, batch_nb, train=False, return_est=True)
            loss_tmp += loss_step
            cnt += 1
            sp += dur_samples

            ###
            if self.current_epoch % 10 == 0:
                targets = gt
                mixture = input
                spec_hat, time_hat = est_step

                
                n_src, n_batch, n_channel, time_length = time_hat.shape
                assert n_batch == 1
                time_hat = time_hat.squeeze(1).permute(0, 2, 1)

                targets = targets[:, :, :, :time_length]
                targets = targets.squeeze(0).permute(0, 2, 1)

                mixture = mixture[:, :, :time_length]
                mixture = mixture.permute(0, 2, 1)
                mix_audio = mixture.clone()
                mixture = mixture.repeat(n_src, 1, 1)
                
                white_noise = 0.0001* torch.randn(targets.shape)
                targets = targets + white_noise
                time_hat = time_hat + white_noise
                
                sdr_mix, _, _, _ = museval.evaluate(targets.detach().cpu(), mixture.detach().cpu())
                sdr, _, _, _ = museval.evaluate(targets.detach().cpu(), time_hat.detach().cpu())
                nan_mask = np.isnan(sdr.reshape(1, -1))
                
                if nan_mask.any():
                    print("GOT NAN")
                    print(targets)
                    print(time_hat)
                    print(sdr.reshape(1,-1))
                
                

                sdr_tmp += np.mean(sdr, axis=1)
                sdri_tmp += np.mean(sdr - sdr_mix, axis=1)

            if batch_tmp[0].shape[-1] < dur_samples or batch[0].shape[-1] == cnt * dur_samples:
                break
        loss = loss_tmp / cnt
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        if self.current_epoch % 10 == 0:
            sdr = sdr_tmp / cnt
            sdri = sdri_tmp / cnt
            for i, src in enumerate(self.model.sources): 
                self.log(f"val_SDR_{src}", sdr[i], on_epoch=True, prog_bar=True)
                self.log(f"val_SDRi_{src}", sdri[i], on_epoch=True, prog_bar=True)

            if batch_nb % 8 == 0:
                valve1_hat = np.array(time_hat[0, :, 0].reshape(-1,1).detach().cpu())
                valve2_hat = np.array(time_hat[1, :, 0].reshape(-1,1).detach().cpu())

                valve1_gt = np.array(targets[0, :, 0].reshape(-1,1).detach().cpu())
                valve2_gt = np.array(targets[1, :, 0].reshape(-1,1).detach().cpu())

                mixture_audio = np.array(mix_audio[0, :, 0].reshape(-1,1).detach().cpu())

                self.logger.experiment.log({
                    "val_valve1": [wandb.Audio(valve1_hat, sample_rate = 16000)],
                    "gt_valve1": [wandb.Audio(valve1_gt, sample_rate = 16000)],
                    "val_valve2": [wandb.Audio(valve2_hat, sample_rate = 16000)],
                    "gt_valve2": [wandb.Audio(valve2_gt, sample_rate = 16000)],
                    "mixture": [wandb.Audio(mixture_audio, sample_rate = 16000)],
                })
                
            self.log("val_mean_SDR", np.mean(sdr), on_epoch=True, prog_bar=True)
            self.log("val_mean_SDRi", np.mean(sdri), on_epoch=True, prog_bar=True)



def main(conf, args):
    # Set seed for random
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # create output dir if not exist
    exp_dir = Path(args.output)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load Datasets
    train_dataset, valid_dataset = dataloader.load_datasets(parser, args)
    dataloader_kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True} if torch.cuda.is_available() else {}
    )
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # Define model and optimizer
    if args.pretrained is not None:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, train_dataset)

    max_bin = bandwidth_to_max_bin(train_dataset.sample_rate, args.in_chan, args.bandwidth)

    if args.use_control:
        x_unmix = XUMXControl(
            window_length=args.window_length,
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            in_chan=args.in_chan,
            n_hop=args.nhop,
            sources=['s1', 's2', 's3', 's4'][:args.num_src_in_mix],  #sources=args.sources,
            max_bin=max_bin,
            bidirectional=args.bidirectional,
            sample_rate=train_dataset.sample_rate,
            spec_power=args.spec_power,
            return_time_signals=True,
        )
    else:
        x_unmix = XUMX(
            window_length=args.window_length,
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            in_chan=args.in_chan,
            n_hop=args.nhop,
            sources=['s1', 's2', 's3', 's4'][:args.num_src_in_mix], #sources=args.sources,

            max_bin=max_bin,
            bidirectional=args.bidirectional,
            sample_rate=train_dataset.sample_rate,
            spec_power=args.spec_power,
            return_time_signals=True,
        )

    optimizer = make_optimizer(
        x_unmix.parameters(), lr=args.lr, optimizer="adam", weight_decay=args.weight_decay
    )

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_gamma, patience=args.lr_decay_patience, cooldown=10
    )

    # Save config
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=True)

    # Define Loss function.
    if args.loss_pit:
        base_loss_func = MultiDomainLoss(
            window_length=args.window_length,
            in_chan=args.in_chan,
            n_hop=args.nhop,
            spec_power=args.spec_power,
            nb_channels=args.nb_channels,
            loss_combine_sources=args.loss_combine_sources,
            loss_use_multidomain=args.loss_use_multidomain,
            mix_coef=args.mix_coef,
            reduce="",
        )
        loss_func = CustomPITLossWrapper(loss_func=base_loss_func, pit_from="perm_avg")
    else:
        loss_func = MultiDomainLoss(
            window_length=args.window_length,
            in_chan=args.in_chan,
            n_hop=args.nhop,
            spec_power=args.spec_power,
            nb_channels=args.nb_channels,
            loss_combine_sources=args.loss_combine_sources,
            loss_use_multidomain=args.loss_use_multidomain,
            mix_coef=args.mix_coef,
            reduce="mean",
        )
    if args.use_control:
        system = XUMXControlManager(
            model=x_unmix,
            loss_func=loss_func,
            optimizer=optimizer,
            train_loader=train_sampler,
            val_loader=valid_sampler,
            scheduler=scheduler,
            config=conf,
            val_dur=args.val_dur,
        )
    else:
        system = XUMXManager(
            model=x_unmix,
            loss_func=loss_func,
            optimizer=optimizer,
            train_loader=train_sampler,
            val_loader=valid_sampler,
            scheduler=scheduler,
            config=conf,
            val_dur=args.val_dur,
        )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", verbose=True, save_top_k=5,
    )
    callbacks.append(checkpoint)
    callbacks.append(es)

    # Don't ask GPU if they are not available.
    run_name = os.path.basename(os.path.normpath(exp_dir))
    wandb.init(save_code=True, config=args, name=run_name)
    wandb_logger = WandbLogger(name=run_name)
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        # limit_train_batches=1.0,  # Useful for fast experiment
        logger = wandb_logger,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_dataset.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    args = parser.parse_args()          # just for conf name
    with open(args.conf) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic, plain_args)
