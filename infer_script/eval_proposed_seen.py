import argparse
import json
import torch
import torch.nn as nn
import os
import librosa
import dac
import warnings
import mir_eval
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from audiotools import AudioSignal
from utils import yaml_config_hook
from evaluate_metrics import MultiScaleSTFTLoss, LogRMSEnvelopeLoss, F0EvalLoss
from dataset import EDM_MN_Test_Dataset
from dac.nn.loss import MelSpectrogramLoss, L1Loss

# Filter out specific warnings
warnings.filterwarnings("ignore", message="stft_data changed shape")
warnings.filterwarnings("ignore", message="Audio amplitude > 1 clipped when saving")

LENGTH = 44100*3
DURATION = 3.0

class EDMFACInference:
    def __init__(
        self,
        checkpoint_path,
        config_path="configs/config.yaml",
        device="cuda",
    ):
        """
        Initialize the EDM-FAC inference model

        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to the configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load configuration
        self.config = yaml_config_hook(config_path)
        self.args = argparse.Namespace(**self.config)

        # Get parameters
        self.sample_rate = self.args.sample_rate
        self.hop_length = self.args.hop_length

        # Initialize model
        self.generator = dac.model.MyDAC(
            encoder_dim=self.args.encoder_dim,
            encoder_rates=self.args.encoder_rates,
            latent_dim=self.args.latent_dim,
            decoder_dim=self.args.decoder_dim,
            decoder_rates=self.args.decoder_rates,
            adsr_enc_dim=self.args.adsr_enc_dim,
            adsr_enc_ver=self.args.adsr_enc_ver,
            sample_rate=self.args.sample_rate,
            timbre_classes=self.args.timbre_classes,
            adsr_classes=self.args.adsr_classes,
            pitch_nums=self.args.max_note - self.args.min_note + 1, # 88
            use_gr_content=self.args.use_gr_content,
            use_gr_adsr=self.args.use_gr_adsr,
            use_gr_timbre=self.args.use_gr_timbre,
            use_FiLM=self.args.use_FiLM,
            rule_based_adsr_folding=self.args.rule_based_adsr_folding,
            use_cross_attn=self.args.use_cross_attn,
        ).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()

        # Load losses for evaluation
        # 1) Multi-scale STFT Loss
        self.stft_loss = MultiScaleSTFTLoss().to(self.device)

        # 2) Envelope L1 Loss (Log-RMS envelope)
        self.envelope_loss = LogRMSEnvelopeLoss().to(self.device)

        # 3) Mel-Spectrogram Loss (match training settings)
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
        ).to(self.device)

        # 4) L1 waveform loss
        self.l1_eval_loss = L1Loss().to(self.device)

        # 5) F0 Evaluation Loss
        # self.f0_eval_loss = F0Evaluator(
        #     ref_hz=440.0,
        #     fmin=50,
        #     fmax=1100,
        #     frame_length=2048,
        #     hop_length=512
        # )#.to(self.device)
        self.f0_eval_loss = F0EvalLoss(
            hop_length=160, #512,
            fmin=50.0,
            fmax=1100.0,
            model_size='tiny', #'full',
            voicing_thresh=0.5,
            weight=1.0,
            device=self.device,
            sr_in=44100,
            sr_out=16000,
            resample_to_16k=True,
        ).to(self.device)

        print(f"EDM-FAC model loaded on {self.device}")


    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'generator_state_dict' in checkpoint:
            # Training checkpoint format with separate state dicts
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded checkpoint from step {checkpoint.get('iter', 'unknown')}")
        elif 'generator' in checkpoint:
            # Alternative format
            self.generator.load_state_dict(checkpoint['generator'])
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        elif 'model_state_dict' in checkpoint:
            # Simple format
            self.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint contains the model state dict directly
            self.generator.load_state_dict(checkpoint)

        print("Model weights loaded successfully")

    def load_audio(self, audio_path):
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file

        Returns:
            AudioSignal object
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.args.sample_rate, mono=True)
        audio = audio[:LENGTH]

        # Convert to AudioSignal
        audio_signal = AudioSignal(torch.tensor(audio).unsqueeze(0).unsqueeze(0), self.args.sample_rate)
        return audio_signal


    @torch.no_grad()
    def evaluate_loader(self, data_loader: DataLoader, output_dir: str, convert_type: str = "reconstruction"):
        os.makedirs(output_dir, exist_ok=True)

        # Aggregators
        overall = {"stft": 0.0, "l1": 0.0, "mel": 0.0, "env": 0.0, "num": 0}

        # Aggregators for MIR melody F0 metrics
        f0_eval_overall = {"f0_corr": 0.0, "f0_rmse": 0.0, "counter": 0, "high_rmse": [], "nan": []}

        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            orig_audio = batch['orig_audio'].to(self.device)
            ref_audio = batch['ref_audio'].to(self.device)
            target_audio = batch['gt_audio'].to(self.device)
            batch['metadata']

            out = self.generator.conversion(
                orig_audio=orig_audio.audio_data,
                ref_audio=None if convert_type == "reconstruction" else ref_audio.audio_data,
                convert_type=convert_type,
            )

            recons = AudioSignal(out["audio"], self.args.sample_rate)
            bs = int(out["audio"].shape[0])

            stft_val = self.stft_loss(recons, target_audio)
            env_val = self.envelope_loss(recons, target_audio)

            overall["stft"] += float(stft_val.item()) * bs
            overall["env"] += float(env_val.item()) * bs
            overall["num"] += bs
            print("STFT", stft_val.item(), "ENV", env_val.item())

            # # Calculate F0 Evaluation Loss
            # f0_summary = self.f0_eval_loss.get_metrics(recons, target_audio, metadata)
            # f0_eval_overall["high_rmse"].extend(f0_summary["high_rmse_paths"])
            # f0_eval_overall["nan"].extend(f0_summary["nan_paths"])

            # f0_rmse = [fr.cpu().numpy() for fr in f0_summary["f0_rmse"]]

            # fr_list = []
            # valid_indices = []

            # cnt = 0
            # for i, fr in enumerate(f0_rmse):
            #     # Check for NaN values in both fc and fr
            #     if fr > 100:
            #         cnt += 1
            #     else:
            #         fr_list.append(fr)
            #         valid_indices.append(i)

            # if len(fr_list) == 0:
            #     continue

            # f0_rmse = np.mean(fr_list)
            # bs -= cnt
            # print(bs, cnt, stft_val.item(), env_val.item())

            # f0_eval_overall["f0_rmse"] += float(f0_rmse) * bs
            # f0_eval_overall["counter"] += bs
            # # f0_eval_overall["high_rmse"].extend(f0_summary["high_rmse_paths"])
            # # f0_eval_overall["nan"].extend(f0_summary["nan_paths"])


        n_all = max(1, overall["num"])
        results = {
            "num_total_samples": overall["num"],
            "overall": {
                "stft_loss": overall["stft"] / n_all,
                # "l1_loss": overall["l1"] / n_all,
                # "mel_loss": overall["mel"] / n_all,
                "envelope_loss": overall["env"] / n_all,
                # "f0_corr": f0_eval_overall["f0_corr"] / f0_eval_overall["counter"],
                "f0_rmse": f0_eval_overall["f0_rmse"] / f0_eval_overall["counter"],
                # "high_rmse": f0_eval_overall["high_rmse"],
                # "nan": f0_eval_overall["nan"],
            },
        }

        # Save metadata immediately here as well
        metadata_path = os.path.join(output_dir, f"metadata_{convert_type}.json")
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=4)

        return results


def main():
    parser = argparse.ArgumentParser(description="EDM-FAC Evaluation on Validation/Test Loader")

    # Arguments
    parser.add_argument("--device", default="cuda:0", help="Device to use for inference")
    parser.add_argument("--bs", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--checkpoint", type=str, default="/home/buffett/nas_data/EDM_FAC_LOG/0804_proposed/ckpt/checkpoint_latest.pt", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config_proposed.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="/home/buffett/nas_data/EDM_FAC_LOG/final_eval/0804_proposed/detail", help="Output directory for results/metadata")

    # F0 Multiprocessing options
    parser.add_argument("--use_f0_multiprocessing", action="store_true", default=True, help="Enable multiprocessing for F0 computation")
    parser.add_argument("--max_f0_workers", type=int, default=8, help="Maximum number of F0 computation workers")
    parser.add_argument("--min_batch_for_multiprocessing", type=int, default=4, help="Minimum batch size to use multiprocessing")

    # Parse initial arguments to get config path
    initial_args, _ = parser.parse_known_args()

    # Load config and add config parameters as arguments
    config = yaml_config_hook(initial_args.config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    # Parse all arguments including config parameters
    args = parser.parse_args()

    # Ibnference Model
    model = EDMFACInference(args.checkpoint, args.config, args.device)

    # Build Evaluation Dataset/Loader from Model Config
    test_dataset_recon = EDM_MN_Test_Dataset(
        root_path=args.root_path,
        duration=DURATION, #3.0,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="eval_seen",
        convert_type="reconstruction",
    )

    test_loader_recon = DataLoader(
        test_dataset_recon,
        shuffle=False,
        batch_size=args.bs, # args.batch_size
        num_workers=16, # args.num_workers
        collate_fn=test_dataset_recon.collate,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset_both = EDM_MN_Test_Dataset(
        root_path=args.root_path,
        duration=DURATION, #3.0,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="eval_seen",
        convert_type="conv_both",
    )

    test_loader_both = DataLoader(
        test_dataset_both,
        shuffle=False,
        batch_size=args.bs, # args.batch_size
        num_workers=16, # args.num_workers
        collate_fn=test_dataset_both.collate,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset_adsr = EDM_MN_Test_Dataset(
        root_path=args.root_path,
        duration=DURATION, #3.0,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        split="eval_seen",
        convert_type="conv_adsr",
    )

    test_loader_adsr = DataLoader(
        test_dataset_adsr,
        shuffle=False,
        batch_size=args.bs, # args.batch_size
        num_workers=16, # args.num_workers
        collate_fn=test_dataset_adsr.collate,
        pin_memory=True,
        drop_last=False,
    )

    # Perform evaluation over loader and save metadata
    # results_recon = model.evaluate_loader(test_loader_recon, args.output_dir, convert_type="reconstruction")
    results_both = model.evaluate_loader(test_loader_both, args.output_dir, convert_type="conv_both")
    # results_adsr = model.evaluate_loader(test_loader_adsr, args.output_dir, convert_type="conv_adsr")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
