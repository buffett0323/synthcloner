import sys
import warnings
import argparse
import os
import torch
warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal
from audiotools import ml
from audiotools.ml.decorators import Tracker, timer, when
from audiotools.core import util
from evaluate_metrics import LogRMSEnvelopeLoss, F0EvalLoss

from dataset import EDM_MN_Val_Total_Dataset
from utils import (
    yaml_config_hook, get_infinite_loader, save_checkpoint, load_checkpoint, log_rms
)
import dac


class Wrapper:
    def __init__(
        self,
        args,
        accelerator,
        val_paired_data,
    ):
        self.disentanglement = args.disentanglement # training
        self.convert_type = args.convert_type # validation

        self.generator = dac.model.MyDAC(
            encoder_dim=args.encoder_dim,
            encoder_rates=args.encoder_rates,
            latent_dim=args.latent_dim,
            decoder_dim=args.decoder_dim,
            decoder_rates=args.decoder_rates,
            adsr_enc_dim=args.adsr_enc_dim,
            adsr_enc_ver=args.adsr_enc_ver,
            sample_rate=args.sample_rate,
            timbre_classes=args.timbre_classes,
            adsr_classes=args.adsr_classes,
            pitch_nums=args.max_note - args.min_note + 1, # 88
            use_gr_content=args.use_gr_content,
            use_gr_adsr=args.use_gr_adsr,
            use_gr_timbre=args.use_gr_timbre,
            use_FiLM=args.use_FiLM,
            rule_based_adsr_folding=args.rule_based_adsr_folding,
            use_cross_attn=args.use_cross_attn,
        ).to(accelerator.device)

        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=args.base_lr)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=0.999996)

        self.discriminator = dac.model.Discriminator().to(accelerator.device)
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=args.base_lr)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=0.999996)

        # Losses
        self.stft_loss = MultiScaleSTFTLoss()#.to(accelerator.device)
        self.envelope_loss = LogRMSEnvelopeLoss()#.to(accelerator.device)

        # Val dataset
        self.val_paired_data = val_paired_data


def main(args, accelerator):
    device = accelerator.device
    util.seed(args.seed)
    print(f"Using device: {device}")

    convert_type = args.conv_type
    print(f"Convert type: {convert_type}")

    val_paired_data = EDM_MN_Val_Total_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path,
        duration=3, #args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split=args.split,
        perturb_content=args.perturb_content,
        perturb_adsr=args.perturb_adsr,
        perturb_timbre=args.perturb_timbre,
        get_midi_only_from_onset=args.get_midi_only_from_onset,
        mask_delay_frames=args.mask_delay_frames,
        disentanglement_mode=args.disentanglement,
        pair_cnts=args.pair_cnts,
        reconstruction=True if convert_type == "reconstruction" else False,
    )

    val_paired_loader = accelerator.prepare_dataloader(
        val_paired_data,
        start_idx=0,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=val_paired_data.collate,
    )
    wrapper = Wrapper(args, accelerator, val_paired_data)
    load_checkpoint(args, device, args.iter, wrapper)

    # Start iteration
    counter = 0

    for i, paired_batch in tqdm(enumerate(val_paired_loader), desc="Evaluating", total=len(val_paired_loader)):
        batch = util.prepare_batch(paired_batch, accelerator.device)

        if convert_type == "reconstruction":
            pass

        else:
            target_audio = batch[f'target_{convert_type}']
            metadata = batch['metadata']
            with torch.no_grad():
                out = wrapper.generator.conversion(
                    orig_audio=batch['orig_audio'].audio_data,
                    ref_audio=batch['ref_audio'].audio_data,
                    convert_type=convert_type,
                )

            recons = AudioSignal(out["audio"], args.sample_rate)

            for i in range(len(metadata)):
                single_orig = AudioSignal(batch['orig_audio'].audio_data[i].cpu(), args.sample_rate)
                single_ref = AudioSignal(batch['ref_audio'].audio_data[i].cpu(), args.sample_rate)
                single_target = AudioSignal(target_audio.audio_data[i].cpu(), args.sample_rate)
                single_recon = AudioSignal(recons.audio_data[i].cpu(), args.sample_rate)

                stft_loss = wrapper.stft_loss(single_recon, single_target)
                envelope_loss = wrapper.envelope_loss(single_recon, single_target)

                if stft_loss < 3 and envelope_loss <= 0.1:
                    counter += 1
                    single_recon.write(f"{args.save_audio_path}/{counter:02d}_{convert_type}.wav")
                    single_target.write(f"{args.save_audio_path}/{counter:02d}_gt.wav")
                    single_orig.write(f"{args.save_audio_path}/{counter:02d}_orig.wav")
                    single_ref.write(f"{args.save_audio_path}/{counter:02d}_ref.wav")


                if counter >= 10:
                    break

            if counter >= 10:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")
    parser.add_argument("--conv_type", default="both")
    parser.add_argument("--pair_cnts", default=10, type=int)
    parser.add_argument("--iter", default=-1, type=int)
    parser.add_argument("--split", default="eval_seen_extreme_adsr") # eval_seen_normal_adsr
    parser.add_argument("--save_audio_path", default="/mnt/gestalt/home/buffett/EDM_FAC_LOG/demo_website/sample_audio")
    config = yaml_config_hook("configs/config_proposed_no_mask.yaml")
    # config = yaml_config_hook("configs/config_proposed_final.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    # Initialize accelerator
    accelerator = ml.Accelerator()
    if accelerator.local_rank != 0:
        sys.tracebacklimit = 0
    main(args, accelerator)
