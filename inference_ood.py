import warnings
import argparse
import os
import glob
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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    util.seed(args.seed)
    print(f"Using device: {device}")

    # Initialize wrapper and load checkpoint
    wrapper = SimpleWrapper(args, device)
    load_checkpoint(args, device, args.iter, wrapper)

    # Set model to evaluation mode
    wrapper.generator.eval()

    # Get all original and reference audio files
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    orig_files = sorted(glob.glob(os.path.join(input_dir, "0*_orig.wav")))
    ref_files = sorted(glob.glob(os.path.join(input_dir, "vital_0*.wav")))

    if not orig_files:
        print(f"No original audio files matching '0*_orig.wav' found in {input_dir}")
        return
    if not ref_files:
        print(f"No reference audio files matching 'vital_0*.wav' found in {input_dir}")
        return

    print(f"Found {len(orig_files)} original files and {len(ref_files)} reference files.")

    # Process all permutations
    for orig_path in orig_files:
        orig_name = os.path.basename(orig_path).replace(".wav", "")
        print(f"Loading original audio: {orig_path}")
        orig_audio_tensor = load_audio(orig_path, args.sample_rate).to(device)

        for ref_path in ref_files:
            ref_name = os.path.basename(ref_path).replace(".wav", "")
            print(f"  Loading reference audio: {ref_path}")
            ref_audio_tensor = load_audio(ref_path, args.sample_rate).to(device)

            # Perform conversion
            print(f"  Performing {args.convert_type} conversion for {orig_name} -> {ref_name}...")
            with torch.no_grad():
                out = wrapper.generator.conversion(
                    orig_audio=orig_audio_tensor,
                    ref_audio=ref_audio_tensor,
                    convert_type=args.convert_type,
                )

            # Get converted audio
            converted_audio = out["audio"]

            # Save output
            output_name = f"{orig_name}_to_{ref_name}_{args.convert_type}.wav"
            output_path = os.path.join(output_dir, output_name)
            save_audio(converted_audio, output_path, args.sample_rate)
            print(f"  Converted audio saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch EDM-FAC Inference")

    # Audio directory argument
    parser.add_argument("--input_dir", default="audios_ood", help="Directory containing audio files")
    parser.add_argument("--iter", default=440000, type=int, help="Checkpoint iteration to load (-1 for latest)")
    parser.add_argument("--output_dir", default="audios_ood_converted", help="Directory to save converted audio files")

    # Load config
    config = yaml_config_hook("configs/config_proposed_final.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.input_dir, exist_ok=True)

    main(args)
