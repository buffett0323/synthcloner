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
from evaluate_metrics import LogRMSEnvelopeLoss

from utils import (
    yaml_config_hook, get_infinite_loader, save_checkpoint, load_checkpoint, log_rms
)
import dac
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict
import soundfile as sf


class EDM_MN_Val_Total_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        duration: float = 1.0,
        sample_rate: int = 44100,
        reconstruction: bool = False,
    ):
        self.root_path = root_path
        self.duration = duration
        self.reconstruction = reconstruction
        self.sample_rate = sample_rate

        # Storing names
        self.paired_data = []
        self.single_data = []
        self._build_paired_index()



    def _build_paired_index(self):
        if self.reconstruction:
            print("Reconstruction mode")
            for counter, data in tqdm(enumerate(self.metadata), desc=f"Building paired index for Multi-Notes {self.split}"):
                wav_path = data['file_path']
                timbre_id = data['timbre_index']
                adsr_id = data['adsr_index']
                midi_id = data['midi_index']

                self.ids_to_item_idx[f"T{timbre_id:03d}_ADSR{adsr_id:03d}_C{midi_id:03d}"] = counter
                self.paired_data.append((
                    timbre_id, midi_id, adsr_id, wav_path
                ))

        else:
            amount = 16
            for i in range(1, amount+1):
                orig_path = f"{self.root_path}/{i:02d}_orig.wav"
                ref_path = f"{self.root_path}/{i:02d}_ref.wav"
                gt_path = f"{self.root_path}/{i:02d}_gt.wav"
                self.paired_data.append(
                    (orig_path, ref_path, gt_path)
                )


    def _load_audio(self, file_path: Path, offset: float = 0.0) -> AudioSignal:
        signal, _ = sf.read(
            file_path,
            start=int(offset*self.sample_rate),
            frames=int(self.duration*self.sample_rate)
        )
        # signal = signal.mean(axis=1, keepdims=False)
        return AudioSignal(signal, self.sample_rate)


    def __len__(self):
        return len(self.paired_data)


    def __getitem__(self, idx):
        if self.reconstruction:
            orig_path, ref_path, gt_path = self.paired_data[idx]
            orig_audio = self._load_audio(orig_path, 0.0)
            return {
                'orig_audio': orig_audio,
                'ref_audio': orig_audio,
                'target_timbre': orig_audio,
                'target_adsr': orig_audio,
                'target_both': orig_audio,

                'metadata': {
                    'id': idx+1,
                }
            }

        else:
            orig_path, ref_path, gt_path = self.paired_data[idx]
            orig_audio = self._load_audio(orig_path, 0.0)
            ref_audio = self._load_audio(ref_path, 0.0)
            gt_audio = self._load_audio(gt_path, 0.0)

            return {
                'orig_audio': orig_audio,
                'ref_audio': ref_audio,
                'target_timbre': gt_audio,
                'target_adsr': gt_audio,
                'target_both': gt_audio,

                'metadata': {
                    'id': idx+1,
                }
            }

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        return {
            'orig_audio': AudioSignal.batch([item['orig_audio'] for item in batch]),
            'ref_audio': AudioSignal.batch([item['ref_audio'] for item in batch]),
            'target_timbre': AudioSignal.batch([item['target_timbre'] for item in batch]),
            'target_adsr': AudioSignal.batch([item['target_adsr'] for item in batch]),
            'target_both': AudioSignal.batch([item['target_both'] for item in batch]),
            'metadata': [item['metadata'] for item in batch]
        }


class Wrapper:
    def __init__(
        self,
        args,
        accelerator,
        val_paired_data,
    ):
        self.disentanglement = args.disentanglement # training
        self.convert_type = args.convert_type # validation

        self.generator = dac.model.ABL_DAC(
            encoder_dim=args.encoder_dim,
            encoder_rates=args.encoder_rates,
            latent_dim=args.latent_dim,
            decoder_dim=args.decoder_dim,
            decoder_rates=args.decoder_rates,
            sample_rate=args.sample_rate,
            timbre_classes=args.timbre_classes,
            pitch_nums=args.max_note - args.min_note + 1, # 88
        ).to(accelerator.device)

        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=args.base_lr)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=0.999996)

        self.discriminator = dac.model.Discriminator().to(accelerator.device)
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=args.base_lr)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=0.999996)

        # Losses
        self.stft_loss = MultiScaleSTFTLoss().to(accelerator.device)
        self.envelope_loss = LogRMSEnvelopeLoss().to(accelerator.device)

        # Val dataset
        self.val_paired_data = val_paired_data



def main(args, accelerator):
    device = accelerator.device
    util.seed(args.seed)
    print(f"Using device: {device}")

    convert_type = args.conv_type
    print(f"Convert type: {convert_type}")

    val_paired_data = EDM_MN_Val_Total_Dataset(
        root_path="/home/buffett/research/demo_synth/audios",
        duration=3, #args.duration,
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
    total_stft_loss = []
    total_envelope_loss = []
    for i, paired_batch in tqdm(enumerate(val_paired_loader), desc="Evaluating", total=len(val_paired_loader)):
        batch = util.prepare_batch(paired_batch, accelerator.device)

        if convert_type == "reconstruction":
            target_audio = batch['orig_audio']
            with torch.no_grad():
                out = wrapper.generator.conversion(
                    orig_audio=batch['orig_audio'].audio_data,
                    ref_audio=None,
                    convert_type=convert_type,
                )

            recons = AudioSignal(out["audio"], args.sample_rate)
            stft_loss = wrapper.stft_loss(recons, target_audio)
            envelope_loss = wrapper.envelope_loss(recons, target_audio)
            total_stft_loss.append(stft_loss)
            total_envelope_loss.append(envelope_loss)
            print(f"Batch {i}: STFT Loss: {stft_loss:.4f}, Envelope Loss: {envelope_loss:.4f}")

        else:
            target_audio = batch[f'target_{convert_type}']
            metadata = batch['metadata']


            with torch.no_grad():
                out = wrapper.generator.conversion(
                    orig_audio=batch['orig_audio'].audio_data,
                    ref_audio=batch['ref_audio'].audio_data,
                    convert_type=convert_type,
                )

            recons = AudioSignal(out["audio"].cpu(), args.sample_rate)

            for i in range(len(metadata)):
                single_recon = AudioSignal(recons.audio_data[i], args.sample_rate)
                single_recon.write(f"{args.save_audio_path}/{metadata[i]['id']:02d}_recon_abl.wav")

            # stft_loss = wrapper.stft_loss(recons, target_audio)
            # envelope_loss = wrapper.envelope_loss(recons, target_audio)
            # total_stft_loss.append(stft_loss)
            # total_envelope_loss.append(envelope_loss)
            # print(f"Batch {i}: STFT Loss: {stft_loss:.4f}, Envelope Loss: {envelope_loss:.4f}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM-FAC")
    parser.add_argument("--conv_type", default="timbre")
    parser.add_argument("--iter", default=400000, type=int)
    parser.add_argument("--split", default="eval_seen_extreme_adsr") # eval_seen_normal_adsr
    parser.add_argument("--save_audio_path", default="/home/buffett/research/demo_synth/audios")
    config = yaml_config_hook("configs/config_mn_ablation.yaml")

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
