import argparse
import json
import torch
import os
import librosa
import numpy as np
import torch.nn as nn
import dac
import warnings

from audiotools import AudioSignal
from audiotools.core import util
from utils import yaml_config_hook
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, L1Loss

# Filter out specific warnings
warnings.filterwarnings("ignore", message="stft_data changed shape")
warnings.filterwarnings("ignore", message="Audio amplitude > 1 clipped when saving")
LENGTH = 44100*3

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
            use_z_gt=self.args.use_z_gt,
        ).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()

        # Load losses for evaluation
        self.stft_loss = MultiScaleSTFTLoss().to(self.device)
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None],
            pow=1.0,
            mag_weight=0.0,
        ).to(self.device)
        self.l1_loss = L1Loss().to(self.device)
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
        audio, sr = librosa.load(audio_path, sr=self.args.sample_rate, mono=True)
        audio = audio[:LENGTH]

        # Convert to AudioSignal
        audio_signal = AudioSignal(torch.tensor(audio).unsqueeze(0).unsqueeze(0), self.args.sample_rate)
        return audio_signal

    @torch.no_grad()
    def convert_audio(self, orig_audio_path, ref_audio_path, gt_audio_path,
                      output_dir, convert_type="timbre", prefix=""):
        """
        Perform audio conversion

        Args:
            orig_audio_path: Path to original audio
            ref_audio_path: Path to reference audio
            output_dir: Directory to save all audio files
            convert_type: Type of conversion ("timbre", "content", "adsr", or "both")
        """
        # Load audio files
        orig_audio = self.load_audio(orig_audio_path)
        ref_audio = self.load_audio(ref_audio_path)
        gt_audio = self.load_audio(gt_audio_path)

        # Move to device
        orig_audio = orig_audio.to(self.device)
        ref_audio = ref_audio.to(self.device)
        gt_audio = gt_audio.to(self.device)

        # Forward pass
        with torch.no_grad():
            out = self.generator.conversion(
                orig_audio=orig_audio.audio_data,
                ref_audio=ref_audio.audio_data,
                convert_type=convert_type,
            )

        # Get converted audio
        converted_audio = AudioSignal(out["audio"].cpu(), self.args.sample_rate)

        # Normalize audio to prevent clipping
        max_amp = torch.max(torch.abs(converted_audio.audio_data))
        if max_amp > 1.0:
            converted_audio.audio_data = converted_audio.audio_data / max_amp * 0.95
            print(f"Audio normalized from max amplitude {max_amp:.3f} to 0.95")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save all audio files
        orig_audio_cpu = AudioSignal(orig_audio.audio_data.cpu(), self.args.sample_rate)
        ref_audio_cpu = AudioSignal(ref_audio.audio_data.cpu(), self.args.sample_rate)
        gt_audio_cpu = AudioSignal(gt_audio.audio_data.cpu(), self.args.sample_rate)

        orig_audio_cpu.write(os.path.join(output_dir, f"{prefix}orig.wav"))
        ref_audio_cpu.write(os.path.join(output_dir, f"{prefix}ref.wav"))
        converted_audio.write(os.path.join(output_dir, f"{prefix}conv_{convert_type}.wav"))
        gt_audio_cpu.write(os.path.join(output_dir, f"{prefix}gt_{convert_type}.wav"))

        # Calculate metrics - ensure all tensors are on the same device
        try:
            # Move both audio signals to the same device as the loss functions
            converted_audio_device = converted_audio.to(self.device)
            gt_audio_device = gt_audio.to(self.device)

            stft_loss = self.stft_loss(converted_audio_device, gt_audio_device)
        except Exception as e:
            print(f"STFT loss calculation failed: {e}")
            stft_loss = torch.tensor(0.0, device=self.device)

        try:
            mel_loss = self.mel_loss(converted_audio_device, gt_audio_device)
        except Exception as e:
            print(f"Mel loss calculation failed: {e}")
            mel_loss = torch.tensor(0.0, device=self.device)

        try:
            l1_loss = self.l1_loss(converted_audio_device, gt_audio_device)
        except Exception as e:
            print(f"L1 loss calculation failed: {e}")
            l1_loss = torch.tensor(0.0, device=self.device)

        # Return additional information
        results = {
            'original_audio_path': orig_audio_path,
            'reference_audio_path': ref_audio_path,
            'ground_truth_audio_path': gt_audio_path,
            'output_dir': output_dir,
            'convert_type': convert_type,
            'stft_loss': stft_loss.item(),
            'mel_loss': mel_loss.item(),
            'l1_loss': l1_loss.item(),
        }

        return results

def main():
    parser = argparse.ArgumentParser(description="EDM-FAC Single Audio Conversion")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--orig_audio", required=True, help="Path to original audio file")
    parser.add_argument("--ref_audio", required=True, help="Path to reference audio file")
    parser.add_argument("--gt_audio", required=True, help="Path to ground truth audio file")
    parser.add_argument("--output_dir", required=True, help="Output directory to save all audio files")
    parser.add_argument("--convert_type", default="timbre", choices=["timbre", "adsr", "both"],
                       help="Type of conversion to perform")
    parser.add_argument("--device", default="cuda", help="Device to use for inference")
    parser.add_argument("--prefix", default="", help="Prefix to add to the output files")

    args = parser.parse_args()

    # Initialize inference model
    model = EDMFACInference(
        args.checkpoint,
        args.config,
        args.device
    )

    # Perform conversion
    results = model.convert_audio(
        args.orig_audio,
        args.ref_audio,
        args.gt_audio,
        args.output_dir,
        args.convert_type,
        args.prefix
    )

    # Save metadata
    metadata_path = os.path.join(args.output_dir, f"{args.prefix}metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print results
    print(f"Conversion completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"Files saved:")
    print(f"  - orig.wav")
    print(f"  - ref_{args.convert_type}.wav")
    print(f"  - conv_{args.convert_type}.wav")
    print(f"  - gt.wav")
    print(f"  - metadata.json")
    print(f"STFT Loss: {results['stft_loss']:.4f}")
    print(f"Mel Loss: {results['mel_loss']:.4f}")
    print(f"L1 Loss: {results['l1_loss']:.4f}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
