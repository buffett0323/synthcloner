import argparse
import json
import torch
import os
import librosa
import random
import numpy as np
import soundfile as sf
import torch.nn as nn
import pretty_midi
import math
import dac
import matplotlib.pyplot as plt
import warnings

from audiotools import AudioSignal
from audiotools.core import util
from pathlib import Path
from utils import yaml_config_hook
from tqdm import tqdm
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss

# Filter out specific warnings
warnings.filterwarnings("ignore", message="stft_data changed shape")
warnings.filterwarnings("ignore", message="Audio amplitude > 1 clipped when saving")

class EDMFACInference:
    def __init__(
        self,
        checkpoint_path,
        config_path="configs/config.yaml",
        audio_length=1.0,
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
        self.audio_length = audio_length

        # Get parameters
        self.sample_rate = self.args.sample_rate
        self.hop_length = self.args.hop_length
        self.min_note = self.args.min_note
        self.max_note = self.args.max_note
        self.n_notes = self.max_note - self.min_note + 1

        # Initialize model
        self.generator = dac.model.MyDAC(
            encoder_dim=self.args.encoder_dim,
            encoder_rates=self.args.encoder_rates,
            latent_dim=self.args.latent_dim,
            decoder_dim=self.args.decoder_dim,
            decoder_rates=self.args.decoder_rates,
            sample_rate=self.args.sample_rate,
            timbre_classes=self.args.timbre_classes,
            pitch_nums=self.args.max_note - self.args.min_note + 1,
        ).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()

        # Load losses
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
        # self.gan_loss = GANLoss(discriminator=self.discriminator).to(self.device)

        self.timbre_loss = nn.CrossEntropyLoss().to(self.device)
        self.content_loss = nn.CrossEntropyLoss().to(self.device)

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



    def load_audio(self, audio_path, target_length=None):
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file
            target_length: Target length in samples (optional)

        Returns:
            AudioSignal object
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.args.sample_rate, mono=True)

        # Pad or trim to target length
        if target_length is not None:
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            elif len(audio) > target_length:
                # Trim to target length
                audio = audio[:target_length]

        # Convert to AudioSignal
        audio_signal = AudioSignal(torch.tensor(audio).unsqueeze(0).unsqueeze(0), self.args.sample_rate)
        return audio_signal


    def get_midi_to_pitch_sequence(self, midi_path: str, duration: float) -> torch.Tensor:
        """Convert MIDI file to pitch sequence tensor"""
        n_frames = math.ceil(duration * self.sample_rate / self.hop_length)

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            pitch_sequence = np.zeros((n_frames, self.n_notes))


            for instrument in pm.instruments:
                for note in instrument.notes:
                    start_frame = int(note.start * self.sample_rate / self.hop_length)
                    end_frame = int(note.end * self.sample_rate / self.hop_length)

                    start_frame = max(0, min(start_frame, n_frames-1))
                    end_frame = max(0, min(end_frame, n_frames-1))

                    note_idx = note.pitch - self.min_note

                    if 0 <= note_idx < self.n_notes:
                        pitch_sequence[start_frame:end_frame+1, note_idx] = 1

            return torch.FloatTensor(pitch_sequence)

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            return torch.zeros((n_frames, self.n_notes))


    def plot_midi_piano_roll(self, midi_path: str, plot_path: str, duration: float):
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        fs = 100
        piano_roll = midi_data.get_piano_roll(fs=fs)

        end_frame = min(piano_roll.shape[1], math.ceil(duration * fs))
        piano_roll = piano_roll[:, :end_frame]

        plt.figure(figsize=(14,6))
        plt.imshow(piano_roll, origin='lower', aspect='auto', cmap='viridis') # 'gray_r
        plt.colorbar(label='Velocity')
        plt.title(f'Piano Roll - {os.path.basename(midi_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('MIDI Note Number')
        plt.grid(True)

        # Add x-axis ticks in seconds
        x_ticks = np.arange(0, end_frame, fs)  # Every second
        x_labels = [f'{i/fs:.1f}' for i in x_ticks]
        plt.xticks(x_ticks, x_labels)

        plt.savefig(plot_path)
        plt.close()


    @torch.no_grad()
    def convert_audio(self, target_audio_path, content_ref_path, timbre_ref_path, output_path,
                      store_ref=False, midi_path=None):
        """
        Perform audio conversion

        Args:
            target_audio_path: Path to target audio (for reconstruction)
            content_ref_path: Path to content reference audio
            timbre_ref_path: Path to timbre reference audio
            output_path: Path to save converted audio
        """
        # Calculate target length based on duration
        if self.audio_length != 0.0:
            target_length = int(self.audio_length * self.args.sample_rate)
        else:
            leng = librosa.get_duration(path=target_audio_path)
            target_length = int(leng * self.args.sample_rate)

        # Load audio files
        target_audio = self.load_audio(target_audio_path, target_length)
        content_ref = self.load_audio(content_ref_path, target_length)
        timbre_ref = self.load_audio(timbre_ref_path, target_length)

        # Move to device
        target_audio = target_audio.to(self.device)
        content_ref = content_ref.to(self.device)
        timbre_ref = timbre_ref.to(self.device)


        # Forward pass
        with torch.no_grad():
            out = self.generator.conversion(
                audio_data=content_ref.audio_data,
                timbre_match=timbre_ref.audio_data,
            )

        # Get predicted timbre and pitch
        if midi_path:
            plot_path = output_path.replace(".wav", ".png")
            duration = target_length / self.args.sample_rate
            target_pitch = self.get_midi_to_pitch_sequence(midi_path, duration)
            target_pitch = target_pitch.to(self.device).unsqueeze(0)
            # Plot Piano Roll
            self.plot_midi_piano_roll(midi_path, plot_path, duration)
        else:
            target_pitch = None

        # Get converted audio
        gt_audio = AudioSignal(target_audio.audio_data.cpu(), self.args.sample_rate)
        converted_audio = AudioSignal(out["audio"].cpu(), self.args.sample_rate)

        # Normalize audio to prevent clipping
        max_amp = torch.max(torch.abs(converted_audio.audio_data))
        if max_amp > 1.0:
            converted_audio.audio_data = converted_audio.audio_data / max_amp * 0.95  # Scale to 95% to avoid clipping
            print(f"Audio normalized from max amplitude {max_amp:.3f} to 0.95")

        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gt_audio.write(output_path.replace(".wav", "_gt.wav"))
        converted_audio.write(output_path.replace(".wav", "_recon.wav"))

        if store_ref:
            content_ref = AudioSignal(content_ref.audio_data.cpu(), self.args.sample_rate)
            timbre_ref = AudioSignal(timbre_ref.audio_data.cpu(), self.args.sample_rate)
            content_ref.write(output_path.replace(".wav", "_content_ref.wav"))
            timbre_ref.write(output_path.replace(".wav", "_timbre_ref.wav"))


        # Calculate metrics
        try:
            stft_loss = self.stft_loss(converted_audio, gt_audio)
        except Exception as e:
            print(f"STFT loss calculation failed: {e}")
            stft_loss = torch.tensor(0.0)

        try:
            mel_loss = self.mel_loss(converted_audio, gt_audio)
        except Exception as e:
            print(f"Mel loss calculation failed: {e}")
            mel_loss = torch.tensor(0.0)

        try:
            l1_loss = self.l1_loss(converted_audio, gt_audio)
        except Exception as e:
            print(f"L1 loss calculation failed: {e}")
            l1_loss = torch.tensor(0.0)

        # timbre_loss = self.timbre_loss(out["pred_timbre_id"], out["target_timbre_id"])
        content_loss = self.content_loss(out["pred_pitch"], target_pitch)

        # Return additional information
        results = {
            'converted_ground_truth_path': target_audio_path,
            'content_ref_path': content_ref_path,
            'timbre_ref_path': timbre_ref_path,
            'vq_commitment_loss': out["vq/commitment_loss"].item(),
            'vq_codebook_loss': out["vq/codebook_loss"].item(),
            'stft_loss': stft_loss.item(),
            'mel_loss': mel_loss.item(),
            'l1_loss': l1_loss.item(),
            'content_loss': content_loss.item(),
            # 'timbre_loss': timbre_loss.item(),
        }

        return results



    @torch.no_grad()
    def batch_convert(self, input_dir, output_dir, midi_dir, midi_list_path, timbre_list_path, amount=2):
        """
        Convert multiple audio files in batch

        Args:
            input_dir: Directory containing input audio files
            content_ref_path: Path to content reference audio
            timbre_ref_path: Path to timbre reference audio
            output_dir: Directory to save converted audio files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get validation data
        with open(midi_list_path, "r") as f:
            validation_midi_names = f.read().splitlines()

        with open(timbre_list_path, "r") as f:
            timbre_names = f.read().splitlines()

        # Get both combinations
        all_combinations = []
        for content_ref_name in validation_midi_names:
            for timbre_ref_name in timbre_names:
                file_path = os.path.join(input_dir, f"{timbre_ref_name}_{content_ref_name}.wav")
                if os.path.exists(file_path):
                    all_combinations.append(file_path)


        # Get random combinations
        random.shuffle(all_combinations)
        audio_files = all_combinations[:amount]
        print(f"Found {len(audio_files)} audio files to convert")


        # Convert audio
        results = {}
        for i, audio_file in tqdm(enumerate(audio_files), desc="Converting audio files"):
            audio_info = audio_file.split("/")[-1].split(".wav")[0].split("_")

            timbre_ref_name = audio_info[0]
            content_ref_name = audio_info[1]
            output_file = output_path / f"sample_{i}.wav" #f"{audio_info[0]}_{audio_info[1]}.wav"

            # Random pich one
            timbre_names_wo_myself = [tn for tn in timbre_names if tn != timbre_ref_name]
            content_names_wo_myself = [cn for cn in validation_midi_names if cn != content_ref_name]

            random_pick_timbre = random.choice(timbre_names_wo_myself)
            random_pick_content = random.choice(content_names_wo_myself)

            content_ref_path = os.path.join(input_dir, f"{random_pick_timbre}_{content_ref_name}.wav")
            timbre_ref_path = os.path.join(input_dir, f"{timbre_ref_name}_{random_pick_content}.wav")

            if os.path.exists(content_ref_path) and os.path.exists(timbre_ref_path):

                try:
                    result = self.convert_audio(
                        target_audio_path=str(audio_file),
                        content_ref_path=content_ref_path,
                        timbre_ref_path=timbre_ref_path,
                        output_path=str(output_file),
                        store_ref=True,
                        midi_path=os.path.join(midi_dir, f"{content_ref_name}.mid")
                    )
                    results[f"sample_{i}"] = result

                except Exception as e:
                    print(f"Error converting {os.path.basename(audio_file)}: {e}")
                    continue

        print(f"Batch conversion completed. {len(results)} files converted successfully.")
        return results


def main():
    parser = argparse.ArgumentParser(description="EDM-FAC Inference")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--audio_length", default=0.0, type=float, help="Conversion length")
    parser.add_argument("--device", default="cuda", help="Device to use for inference")
    parser.add_argument("--output_dir", default="sample_audio/", help="Output directory")
    parser.add_argument("--input_dir", help="Input directory for batch mode")
    parser.add_argument("--midi_dir", default="/home/buffett/dataset/EDM_FAC_DATA/single_note_midi/evaluation/midi/", help="MIDI directory for batch mode")
    parser.add_argument("--midi_list_path", default="info/midi_names_mixed_evaluation.txt", help="MIDI list path for batch mode")
    parser.add_argument("--timbre_list_path", default="info/timbre_names_mixed.txt", help="Timbre list path for batch mode")
    parser.add_argument("--mode", default="single_convert", help="Mode to run")
    parser.add_argument("--amount", default=2, type=int, help="Amount of files to convert")

    # # Audio paths
    # parser.add_argument("--target_audio", help="Path to target audio")
    # parser.add_argument("--content_ref", help="Path to content reference audio")
    # parser.add_argument("--timbre_ref", help="Path to timbre reference audio")
    # parser.add_argument("--output", required=True, help="Output path")


    args = parser.parse_args()

    # Initialize inference model
    model = EDMFACInference(
        args.checkpoint,
        args.config,
        args.audio_length,
        args.device
    )


    # Conversion
    if args.mode == "batch_convert":
        results = model.batch_convert(
            args.input_dir,
            args.output_dir,
            args.midi_dir,
            args.midi_list_path,
            args.timbre_list_path,
            args.amount
        )

        # write results to json
        if results:
            with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
                json.dump(results, f, indent=4)
        else:
            print("No results to write")


    elif args.mode == "single_convert":
        results = model.convert_audio(
            args.target_audio,
            args.content_ref,
            args.timbre_ref,
            args.output,
        )




if __name__ == "__main__":
    main()
