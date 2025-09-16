import sys
import warnings
import argparse
import os
import json
import time
import torch
import logging
warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss
from audiotools import AudioSignal
from audiotools import ml
from audiotools.ml.decorators import Tracker, timer, when
from audiotools.core import util

from dataset import EDM_MN_Dataset, EDM_MN_Val_Dataset
from utils import (
    yaml_config_hook, get_infinite_loader, save_checkpoint, load_checkpoint, log_rms
)
import dac


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        self.stft_loss = MultiScaleSTFTLoss().to(accelerator.device)
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
        ).to(accelerator.device)
        self.l1_loss = L1Loss().to(accelerator.device)
        self.gan_loss = GANLoss(discriminator=self.discriminator).to(accelerator.device)

        # Predictor losses
        self.timbre_loss = nn.CrossEntropyLoss().to(accelerator.device)
        self.adsr_loss = nn.CrossEntropyLoss().to(accelerator.device)
        if args.get_midi_only_from_onset:
            self.content_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0)).to(accelerator.device)
        else:
            self.content_loss = nn.BCEWithLogitsLoss().to(accelerator.device)

        # Gradient reversal losses
        if args.use_gr_content:
            self.rev_content_loss = nn.BCEWithLogitsLoss().to(accelerator.device)
        if args.use_gr_adsr:
            self.rev_adsr_loss = nn.CrossEntropyLoss().to(accelerator.device)
        if args.use_gr_timbre:
            self.rev_timbre_loss = nn.CrossEntropyLoss().to(accelerator.device)

        # Envelope loss
        if args.use_env_loss:
            self.env_loss = nn.L1Loss(reduction='mean').to(accelerator.device) # Designed for Tensor

        # Loss lambda parameters
        self.params = {
            "gen/mel-loss": 15.0,
            # "gen/l1-loss": 15.0,

            "adv/loss_feature": 2.0,
            "adv/loss_g": 1.0,

            "vq/commitment_loss": 0.25,
            "vq/codebook_loss": 1.0,

            "pred/timbre_loss": 5.0,
            "pred/content_loss": 5.0,
            "pred/adsr_loss": 5.0, # 1.0,
        }

        # Gradient Reversal Losses
        if args.use_gr_content:
            self.params["rev/content_loss"] = 5.0
        if args.use_gr_adsr:
            self.params["rev/adsr_loss"] = 5.0
        if args.use_gr_timbre:
            self.params["rev/timbre_loss"] = 5.0

        # Other Losses
        if args.use_env_loss:
            self.params["gen/env-loss"] = 10.0

        # Val dataset
        self.val_paired_data = val_paired_data

        # Print params
        print(json.dumps({k: v for k, v in sorted(self.params.items())}, indent=2))


    @staticmethod
    def supervised_acc(pred_logits, target_labels):
        """
        pred_logits: [B, num_classes]
        target_labels: [B]
        """
        preds = torch.argmax(pred_logits, dim=-1)
        correct = (preds == target_labels).sum().item()
        total = target_labels.size(0)
        acc = correct / total
        return acc


@torch.no_grad()
def save_samples(args, accelerator, tracker_step, wrapper):
    wrapper.generator.eval()
    samples = [wrapper.val_paired_data[idx] for idx in args.val_idx]
    batch = wrapper.val_paired_data.collate(samples)
    batch = util.prepare_batch(batch, accelerator.device)

    # Get original & reference audio
    orig_audio = AudioSignal(batch['orig_audio'].audio_data.cpu(), args.sample_rate)
    ref_audio = AudioSignal(batch['ref_audio'].audio_data.cpu(), args.sample_rate)

    os.makedirs(os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}'), exist_ok=True)


    for conv_type in wrapper.convert_type:
        os.makedirs(os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', f'conv_{conv_type}'), exist_ok=True)
        out = wrapper.generator.conversion(
            orig_audio=batch['orig_audio'].audio_data,
            ref_audio=batch['ref_audio'].audio_data,
            # content_match=batch['content_match'].audio_data,
            convert_type=conv_type,
        )

        recons = AudioSignal(out["audio"].cpu(), args.sample_rate)
        recons_gt = AudioSignal(batch[f'target_{conv_type}'].audio_data.cpu(), args.sample_rate)


        # Conversion & Save
        for i, sample_idx in enumerate(args.val_idx):
            single_recon = AudioSignal(recons.audio_data[i], args.sample_rate)
            single_recon_gt = AudioSignal(recons_gt.audio_data[i], args.sample_rate)
            single_orig = AudioSignal(orig_audio.audio_data[i], args.sample_rate)
            single_ref = AudioSignal(ref_audio.audio_data[i], args.sample_rate)


            recon_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', f'conv_{conv_type}', f'{sample_idx:02d}_recon.wav')
            recon_gt_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', f'conv_{conv_type}', f'{sample_idx:02d}_gt.wav')
            orig_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', f'conv_{conv_type}', f'{sample_idx:02d}_orig.wav')
            ref_path = os.path.join(args.save_path, 'sample_audio', f'iter_{tracker_step}', f'conv_{conv_type}', f'{sample_idx:02d}_ref.wav')

            single_recon.write(recon_path)
            single_recon_gt.write(recon_gt_path)
            single_orig.write(orig_path)
            single_ref.write(ref_path)


def main(args, accelerator):

    device = accelerator.device
    util.seed(args.seed)
    print(f"Using device: {device}")

    # Save args.note to save_path/note.txt
    note_path = os.path.join(args.save_path, 'note.txt')
    with open(note_path, 'w') as f:
        f.write(str(args.note) if getattr(args, 'note', None) is not None else '')

    # Checkpoint direction
    os.makedirs(args.ckpt_path, exist_ok=True)


    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(args.save_path, 'sample_audio')).mkdir(exist_ok=True, parents=True)
    tracker = Tracker(
        writer=(
            SummaryWriter(log_dir=f"{args.save_path}/logs")
                if accelerator.local_rank == 0 else None
        ),
        log_file=f"{args.save_path}/log.txt",
        rank=accelerator.local_rank,
    )

    # Build datasets and dataloaders
    train_paired_data = EDM_MN_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="train",
        perturb_content=args.perturb_content,
        perturb_adsr=args.perturb_adsr,
        perturb_timbre=args.perturb_timbre,
        get_midi_only_from_onset=args.get_midi_only_from_onset,
        mask_delay_frames=args.mask_delay_frames,
        mask_prob=args.mask_prob,
        disentanglement_mode=args.disentanglement,
    )

    val_paired_data = EDM_MN_Val_Dataset(
        root_path=args.root_path,
        midi_path=args.midi_path,
        duration=args.duration,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="eval_seen_normal_adsr",
        perturb_content=args.perturb_content,
        perturb_adsr=args.perturb_adsr,
        perturb_timbre=args.perturb_timbre,
        get_midi_only_from_onset=args.get_midi_only_from_onset,
        mask_delay_frames=args.mask_delay_frames,
        disentanglement_mode=args.disentanglement,
    )

    wrapper = Wrapper(args, accelerator, val_paired_data)


    # Load checkpoint if exists
    if args.resume:
        start_iter = load_checkpoint(args, device, -1, wrapper) or 0
        tracker.step = start_iter
        print(f"Resuming from iteration {start_iter}")
    else:
        tracker.step = 0


    # Accelerate dataloaders
    train_paired_loader = accelerator.prepare_dataloader(
        train_paired_data,
        start_idx=tracker.step * args.batch_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=train_paired_data.collate,
    )
    train_paired_loader = get_infinite_loader(train_paired_loader)

    val_paired_loader = accelerator.prepare_dataloader(
        val_paired_data,
        start_idx=0,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=val_paired_data.collate,
    )


    # Trackers settings
    global train_step, validate, save_checkpoint, save_samples
    train_step = tracker.log("Train", "value", history=False)(
        tracker.track("Train", args.num_iters, completed=tracker.step)(train_step)
    )
    validate = tracker.log("Validation", "value", history=False)(
        tracker.track("Validation",
                    int(args.num_iters / args.validate_interval),
                    completed=int(tracker.step / args.validate_interval))(validate)
    )
    save_checkpoint = when(lambda: accelerator.local_rank == 0)(save_checkpoint)
    save_samples = when(lambda: accelerator.local_rank == 0)(save_samples)


    # Loop
    with tracker.live:
        for tracker.step, batch in enumerate(train_paired_loader, start=tracker.step):
            train_step(args, accelerator, batch, wrapper, tracker.step)

            # Save Checkpoint
            if tracker.step % args.save_interval == 0:
                save_checkpoint(args, tracker.step, wrapper)

            # Validation
            if tracker.step % args.validate_interval == 0:
                validate(args, accelerator, val_paired_loader, wrapper)

            # Save validation samples
            if tracker.step % args.sample_freq == 0:
                save_samples(args, accelerator, tracker.step, wrapper)

            if tracker.step == args.num_iters:
                break


# @timer
@torch.no_grad()
def validate(args, accelerator, val_paired_loader, wrapper):
    output = {}

    for i, paired_batch in enumerate(val_paired_loader):
        for conv_type in wrapper.convert_type:
            output.update(validate_step(args, accelerator, paired_batch, wrapper, conv_type))
        if i >= args.validate_steps:
            break

    if hasattr(wrapper.optimizer_g, "consolidate_state_dict"):
        wrapper.optimizer_g.consolidate_state_dict()
        wrapper.optimizer_d.consolidate_state_dict()

    return output



# @timer
@torch.no_grad()
def validate_step(args, accelerator, batch, wrapper, conv_type):
    wrapper.generator.eval()
    wrapper.discriminator.eval()
    batch = util.prepare_batch(batch, accelerator.device)

    target_audio = batch[f'target_{conv_type}']
    with torch.no_grad():
        out = wrapper.generator.conversion(
            orig_audio=batch['orig_audio'].audio_data,
            ref_audio=batch['ref_audio'].audio_data,
            convert_type=conv_type,
        )
    output = {}
    recons = AudioSignal(out["audio"], args.sample_rate)

    # Output Loss
    output[f"gen/stft-loss_{conv_type}"] = wrapper.stft_loss(recons, target_audio)
    output["gen/l1-loss"] = wrapper.l1_loss(recons, target_audio)
    output["gen/mel-loss"] = wrapper.mel_loss(recons, target_audio)

    # Envelope Loss
    if args.use_env_loss:
        recons_env = log_rms(recons.audio_data, hop=args.hop_length)
        target_env = log_rms(target_audio.audio_data, hop=args.hop_length)
        output["gen/env-loss"] = wrapper.env_loss(recons_env, target_env)

    # Timbre prediction loss and accuracy
    pitch_gt = batch['ref_pitch'] if conv_type == "content" else batch['orig_pitch']
    # timbre_gt = batch['ref_timbre'] if conv_type in ["timbre", "both"] else batch['orig_timbre']

    output["pred/content_loss"] = wrapper.content_loss(out["pred_pitch"], pitch_gt)

    timbre_id_gt = batch['ref_timbre'] if conv_type in ["timbre", "both"] else batch['orig_timbre']
    output["pred/timbre_acc"] = wrapper.supervised_acc(out["pred_timbre_id"], timbre_id_gt)

    adsr_id_gt = batch['ref_adsr'] if conv_type in ["adsr", "both"] else batch['orig_adsr']
    output["pred/adsr_acc"] = wrapper.supervised_acc(out["pred_adsr_id"], adsr_id_gt)

    return {k: v for k, v in sorted(output.items())}



# @timer
def train_step(args, accelerator, batch, wrapper, current_iter):
    output = {}
    output.update(train_step_paired(args, accelerator, batch, wrapper, current_iter))
    return output


# @timer
def train_step_paired(args, accelerator, batch, wrapper, current_iter):
    train_start_time = time.time()
    wrapper.generator.train()

    # Only train discriminator after discriminator_iter_start
    if current_iter >= args.discriminator_iter_start:
        wrapper.discriminator.train()
    else:
        wrapper.discriminator.eval()

    # Pre-settings
    output = {}
    batch = util.prepare_batch(batch, accelerator.device)
    with torch.no_grad():
        target_audio = batch['target']
        content_match_data = batch['content_match'] # Masked Content
        timbre_match_data = batch['timbre_match']
        adsr_match_data = batch['adsr_match']

        timbre_id = batch['timbre_id']
        adsr_id = batch['adsr_id']
        pitch = batch['pitch']

    # DAC Model
    with accelerator.autocast():
        out = wrapper.generator(
            audio_data=target_audio.audio_data,
            content_match=content_match_data.audio_data,
            timbre_match=timbre_match_data.audio_data,
            adsr_match=adsr_match_data.audio_data,
        )

        recons = AudioSignal(out["audio"], args.sample_rate)

        # Discriminator Losses - only compute and train if past discriminator_iter_start
        if current_iter >= args.discriminator_iter_start:
            output["adv/disc_loss"] = wrapper.gan_loss.discriminator_loss(recons, target_audio)

            wrapper.optimizer_d.zero_grad()
            accelerator.backward(output["adv/disc_loss"])
            accelerator.scaler.unscale_(wrapper.optimizer_d)
            torch.nn.utils.clip_grad_norm_(wrapper.discriminator.parameters(), 10.0)
            accelerator.step(wrapper.optimizer_d)
            wrapper.scheduler_d.step()
        else:
            # Set discriminator loss to 0 when not training discriminator
            output["adv/disc_loss"] = torch.tensor(0.0, device=accelerator.device)

    # Generator Losses
    with accelerator.autocast():
        output["gen/stft-loss"] = wrapper.stft_loss(recons, target_audio)
        output["gen/mel-loss"] = wrapper.mel_loss(recons, target_audio)
        output["gen/l1-loss"] = wrapper.l1_loss(recons, target_audio)

        # Envelope Loss
        if args.use_env_loss:
            recons_env = log_rms(recons.audio_data, hop=args.hop_length)
            target_env = log_rms(target_audio.audio_data, hop=args.hop_length)
            output["gen/env-loss"] = wrapper.env_loss(recons_env, target_env)

        # Only compute adversarial losses if discriminator is being trained
        if current_iter >= args.discriminator_iter_start:
            output["adv/loss_g"], output["adv/loss_feature"] = wrapper.gan_loss.generator_loss(recons, target_audio)
        else:
            # Set adversarial losses to 0 when discriminator is not being trained
            output["adv/loss_g"] = torch.tensor(0.0, device=accelerator.device)
            output["adv/loss_feature"] = torch.tensor(0.0, device=accelerator.device)

        output["vq/commitment_loss"] = out["vq/commitment_loss"]
        output["vq/codebook_loss"] = out["vq/codebook_loss"]

        # Added predictor losses
        output["pred/timbre_loss"] = wrapper.timbre_loss(out["pred_timbre_id"], timbre_id)
        output["pred/content_loss"] = wrapper.content_loss(out["pred_pitch"], pitch)
        output["pred/adsr_loss"] = wrapper.adsr_loss(out["pred_adsr_id"], adsr_id)

        # Total Loss
        output["loss_gen_all"] = sum([v * output[k] for k, v in wrapper.params.items() if k in output])

    # Optimizer
    wrapper.optimizer_g.zero_grad()
    accelerator.backward(output["loss_gen_all"])
    accelerator.scaler.unscale_(wrapper.optimizer_g)
    torch.nn.utils.clip_grad_norm_(wrapper.generator.parameters(), 1e3)
    accelerator.step(wrapper.optimizer_g)
    wrapper.scheduler_g.step()
    accelerator.update()

    # Logging
    output["time/per_step"] = time.time() - train_start_time

    return {k: v for k, v in sorted(output.items())}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")
    config = yaml_config_hook("configs/config_proposed_final.yaml")

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
