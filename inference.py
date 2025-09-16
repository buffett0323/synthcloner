import sys
import warnings
import argparse
import os
import torch
import soundfile as sf
import numpy as np
warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from evaluate_metrics import LogRMSEnvelopeLoss
from utils import yaml_config_hook, load_checkpoint
import dac


class SimpleWrapper:
    def __init__(self, args, device):
        self.device = device

        # Initialize generator
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
            pitch_nums=args.max_note - args.min_note + 1,
            use_gr_content=args.use_gr_content,
            use_gr_adsr=args.use_gr_adsr,
            use_gr_timbre=args.use_gr_timbre,
            use_FiLM=args.use_FiLM,
            rule_based_adsr_folding=args.rule_based_adsr_folding,
            use_cross_attn=args.use_cross_attn,
        ).to(device)

        # Initialize optimizer and scheduler (needed for checkpoint loading)
        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=args.base_lr)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=1.0)

        # Initialize discriminator (needed for checkpoint loading)
        self.discriminator = dac.model.Discriminator().to(device)
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=args.base_lr)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=1.0)

        # # Losses for evaluation
        # self.stft_loss = MultiScaleSTFTLoss().to(device)
        # self.envelope_loss = LogRMSEnvelopeLoss().to(device)


def load_audio(file_path, sample_rate=44100):
    """Load audio file and return as torch tensor"""
    audio, sr = sf.read(file_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if necessary
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    # Convert to torch tensor and add batch and channel dimensions
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]

    return audio_tensor


def save_audio(audio_tensor, file_path, sample_rate=44100):
    """Save audio tensor to file"""
    # Remove batch and channel dimensions
    audio_np = audio_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # Ensure audio is in valid range
    audio_np = np.clip(audio_np, -1.0, 1.0)

    sf.write(file_path, audio_np, sample_rate)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    util.seed(args.seed)
    print(f"Using device: {device}")

    # Load audio files
    print(f"Loading original audio: {args.orig_audio}")
    orig_audio_tensor = load_audio(args.orig_audio, args.sample_rate)

    ref_audio_tensor = None
    if args.ref_audio:
        print(f"Loading reference audio: {args.ref_audio}")
        ref_audio_tensor = load_audio(args.ref_audio, args.sample_rate)

    # Initialize wrapper and load checkpoint
    wrapper = SimpleWrapper(args, device)
    load_checkpoint(args, device, args.iter, wrapper)

    # Set model to evaluation mode
    wrapper.generator.eval()

    # Perform conversion
    print(f"Performing {args.convert_type} conversion...")
    with torch.no_grad():
        out = wrapper.generator.conversion(
            orig_audio=orig_audio_tensor.to(device),
            ref_audio=ref_audio_tensor.to(device) if ref_audio_tensor is not None else None,
            convert_type=args.convert_type,
        )

    # Get converted audio
    converted_audio = out["audio"]

    # Save output
    output_path = args.output_path or f"converted_{args.convert_type}.wav"
    save_audio(converted_audio, output_path, args.sample_rate)
    print(f"Converted audio saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple EDM-FAC Inference")

    # Audio file arguments
    parser.add_argument("--orig_audio", required=True, help="Path to original audio file")
    parser.add_argument("--ref_audio", help="Path to reference audio file (optional for reconstruction)")
    parser.add_argument("--output_path", help="Path to save converted audio (default: converted_{convert_type}.wav)")

    # Conversion arguments
    parser.add_argument("--convert_type", default="timbre", choices=["reconstruction", "timbre", "adsr", "both"],
                       help="Type of conversion to perform")
    parser.add_argument("--iter", default=400000, type=int, help="Checkpoint iteration to load (-1 for latest)")
    parser.add_argument("--ckpt_path", help="Path to checkpoint directory (overrides config)")

    # Load config
    config = yaml_config_hook("configs/config_proposed_no_ca.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Override checkpoint path if provided
    if args.ckpt_path:
        args.ckpt_path = args.ckpt_path

    # Validate arguments
    if not os.path.exists(args.orig_audio):
        print(f"Error: Original audio file not found: {args.orig_audio}")
        sys.exit(1)

    if args.ref_audio and not os.path.exists(args.ref_audio):
        print(f"Error: Reference audio file not found: {args.ref_audio}")
        sys.exit(1)

    if args.convert_type != "reconstruction" and not args.ref_audio:
        print(f"Error: Reference audio is required for {args.convert_type} conversion")
        sys.exit(1)

    # Create output directory if needed
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    main(args)
