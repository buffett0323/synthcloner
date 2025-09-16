import os
import random
import librosa
import numpy as np
import scipy.signal
import pretty_midi
import torch
import torch.nn.functional as F
from audiotools import AudioSignal

from omegaconf import OmegaConf
from tqdm import tqdm

SAMPLE_RATE = 44100
FILTER_TIME = 4.0


def yaml_config_hook(config_file):
    """
    Load YAML with OmegaConf to support ${variable} interpolation.
    Also supports nested includes via a 'defaults' section.
    """
    # Load main config
    cfg = OmegaConf.load(config_file)

    # Load nested defaults if any (like Hydra-style)
    if "defaults" in cfg:
        for d in cfg.defaults:
            config_dir, cf = d.popitem()
            cf_path = os.path.join(os.path.dirname(config_file), config_dir, f"{cf}.yaml")
            nested_cfg = OmegaConf.load(cf_path)
            cfg = OmegaConf.merge(cfg, nested_cfg)

        del cfg.defaults

    return cfg



def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch



def save_checkpoint(args, iter, wrapper):
    """Save model checkpoint and optimizer state"""
    checkpoint_path = os.path.join(args.ckpt_path, f'checkpoint_{iter}.pt')

    # Save generator
    torch.save({
        'generator_state_dict': wrapper.generator.state_dict(),
        'optimizer_g_state_dict': wrapper.optimizer_g.state_dict(),
        'scheduler_g_state_dict': wrapper.scheduler_g.state_dict(),
        'discriminator_state_dict': wrapper.discriminator.state_dict(),
        'optimizer_d_state_dict': wrapper.optimizer_d.state_dict(),
        'scheduler_d_state_dict': wrapper.scheduler_d.state_dict(),
        'iter': iter
    }, checkpoint_path)

    # Save latest checkpoint by creating a symlink
    latest_path = os.path.join(args.ckpt_path, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(checkpoint_path, latest_path)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(args, device, iter, wrapper):
    """Load model checkpoint and optimizer state"""
    if iter == -1:
        # Load latest checkpoint
        checkpoint_path = os.path.join(args.ckpt_path, 'checkpoint_latest.pt')
    else:
        # Load specific checkpoint
        checkpoint_path = os.path.join(args.ckpt_path, f'checkpoint_{iter}.pt')

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load generator
    wrapper.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)

    # Try to load optimizer states using safe loading
    if safe_load_optimizer_state(wrapper.optimizer_g, checkpoint['optimizer_g_state_dict']):
        print("Successfully loaded generator optimizer state")
    else:
        print("Continuing with fresh generator optimizer state...")

    try:
        wrapper.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        print("Successfully loaded generator scheduler state")
    except (ValueError, KeyError) as e:
        print(f"Warning: Could not load generator scheduler state: {e}")
        print("Continuing with fresh scheduler state...")

    # Load discriminator
    wrapper.discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)

    if safe_load_optimizer_state(wrapper.optimizer_d, checkpoint['optimizer_d_state_dict']):
        print("Successfully loaded discriminator optimizer state")
    else:
        print("Continuing with fresh discriminator optimizer state...")

    try:
        wrapper.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        print("Successfully loaded discriminator scheduler state")
    except (ValueError, KeyError) as e:
        print(f"Warning: Could not load discriminator scheduler state: {e}")
        print("Continuing with fresh scheduler state...")

    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint['iter']



def get_timbre_names(path):
    timbres = set()
    for file in tqdm(os.listdir(path)):
        if file.endswith(".wav"):
            tmp_timbre = file.split(".wav")[0].split("_")
            tmp_timbre = "_".join(tmp_timbre[:-1])
            timbres.add(tmp_timbre)

    timbres = list(timbres)
    random.shuffle(timbres)
    with open("info/timbre_names_mixed.txt", "w") as f:
        for timbre in timbres:
            f.write(timbre + "\n")

    print(len(timbres))


def get_midi_names(path, split):
    midis = set()
    for file in tqdm(os.listdir(path)):
        if file.endswith(".wav"):
            midis.add(file.split(".wav")[0].split("_")[-1])

    midis = list(midis)
    random.shuffle(midis)
    with open(f"info/midi_names_mixed_{split}.txt", "w") as f:
        for midi in midis:
            f.write(midi + "\n")

    print(len(midis))



def process_file(file_info, path):
    stem, file = file_info
    file_name = file.split(".wav")[0]
    if file.endswith(".wav"):
        audio, _ = librosa.load(os.path.join(path, stem, file), sr=None)
        audio = audio[:int(FILTER_TIME * SAMPLE_RATE)]

        # Find indices where amplitude exceeds threshold
        envelope = np.abs(audio)
        peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)
        if len(peaks) == 0: return (file_name, [])

        # Get peak amplitudes
        peak_amplitudes = envelope[peaks]

        # Get top 10 peaks by amplitude
        if len(peaks) > 10:
            top_indices = np.argsort(peak_amplitudes)[-10:]
            peaks = peaks[top_indices]
            peak_amplitudes = peak_amplitudes[top_indices]

        # Convert peak indices to time and filter peaks > 8 seconds
        peak_info = [peak_idx / SAMPLE_RATE for peak_idx in peaks]

        return (file_name, peak_info) if peak_info else (file_name, [])



def get_onset_from_midi(midi_file):
    """
    Extract onset times from a MIDI file, filtering to only include onsets before 8 seconds.

    Args:
        midi_file: Path to the MIDI file

    Returns:
        onset_times: List of onset times in seconds before 8 seconds
    """
    try:
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        # Collect all note onset times
        onset_times = []
        n_count = 0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                n_count += 1
                if note.start <= FILTER_TIME:
                    onset_times.append(note.start)

        return sorted(list(set(onset_times)))

    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return []



def get_top_5_peak_position(audio):
    envelope = np.abs(audio)
    peaks, _ = scipy.signal.find_peaks(envelope, distance=SAMPLE_RATE // 20)
    if len(peaks) == 0: return []

    # Get peak amplitudes
    peak_amplitudes = envelope[peaks]

    # Get top 5 peaks by amplitude
    if len(peaks) > 5:
        top_indices = np.argsort(peak_amplitudes)[-5:]
        peaks = peaks[top_indices]

    # Convert peak indices to time in seconds
    peak_times = peaks / SAMPLE_RATE

    # Return as list for JSON compatibility
    return peak_times.tolist()


def process_beatport_file(file_info):
    """
    Process a single beatport file to extract peak positions

    Args:
        file_info: tuple of (file_path, file_name)

    Returns:
        tuple of (file_name, peaks_list)
    """
    file_path, file_name = file_info
    try:
        audio, _ = librosa.load(file_path, sr=None)
        audio = audio[:int(FILTER_TIME * SAMPLE_RATE)]
        peaks = get_top_5_peak_position(audio)
        return (file_name, peaks)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return (file_name, [])


def safe_load_optimizer_state(optimizer, state_dict):
    """
    Safely load optimizer state dict by matching parameters by name
    rather than by parameter group structure.
    """
    try:
        # First try the standard loading method
        optimizer.load_state_dict(state_dict)
        return True
    except (ValueError, KeyError) as e:
        print(f"Standard optimizer loading failed: {e}")
        print("Attempting parameter-wise loading...")

        # Get current parameter names
        current_params = []
        for group in optimizer.param_groups:
            for param in group['params']:
                current_params.append(param)

        # Create mapping from parameter id to parameter
        if 'state' in state_dict:
            saved_state = state_dict['state']
            current_state = optimizer.state

            # Try to match saved states to current parameters
            # This is a simplified approach - in practice you might want more sophisticated matching
            print("Attempting to restore optimizer state for matching parameters...")
            matched_params = 0

            for param_id, param_state in saved_state.items():
                if param_id < len(current_params):
                    current_state[current_params[param_id]] = param_state
                    matched_params += 1

            print(f"Restored state for {matched_params} parameters")

        return False


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameter_count(num_params):
    """Format parameter count in readable format (e.g., 6M, 300M, 4B)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    else:
        return str(num_params)


def print_model_info(wrapper):
    """Print information about model parameters."""
    gen_params = count_parameters(wrapper.generator)
    disc_params = count_parameters(wrapper.discriminator)
    total_params = gen_params + disc_params

    print(f"\n{'='*50}")
    print(f"MODEL PARAMETER COUNT")
    print(f"{'='*50}")
    print(f"Generator parameters: {format_parameter_count(gen_params)} ({gen_params:,})")
    print(f"Discriminator parameters: {format_parameter_count(disc_params)} ({disc_params:,})")
    print(f"Total parameters: {format_parameter_count(total_params)} ({total_params:,})")
    print(f"{'='*50}\n")



def log_rms(wav, hop=512, eps=1e-7):
    """Return log-RMS envelope. wav: (B, 1, T_samples)."""
    rms = torch.sqrt(
        F.avg_pool1d(wav ** 2, kernel_size=hop, stride=hop) + eps
    )
    return torch.log(rms + eps).squeeze(1)        # (B, T_frames)



def extract_f0_from_audio(audio_signal, sample_rate=44100, fmin=50, fmax=2000):
    """
    Extract F0 from audio signal using librosa.pyin

    Args:
        audio_signal: AudioSignal object or tensor
        sample_rate: Sample rate of the audio
        fmin: Minimum frequency for F0 detection
        fmax: Maximum frequency for F0 detection

    Returns:
        f0: F0 values in Hz, with NaN for unvoiced frames
    """
    if isinstance(audio_signal, AudioSignal):
        audio_data = audio_signal.audio_data.squeeze().cpu().numpy()
    elif isinstance(audio_signal, torch.Tensor):
        audio_data = audio_signal.squeeze().cpu().numpy()
    else:
        audio_data = audio_signal

    # Ensure audio is mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=0)

    # Extract F0 using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=512  # Use same hop length as the model
    )

    return f0

def calculate_f0_error_rate(f0_pred, f0_target, threshold=0.5):
    """
    Calculate F0 error rate between predicted and target F0

    Args:
        f0_pred: Predicted F0 values
        f0_target: Target F0 values
        threshold: Threshold for considering frames as voiced (in Hz)

    Returns:
        error_rate: Percentage of frames with F0 error
        mean_error: Mean absolute F0 error in Hz
    """
    # Convert to numpy arrays
    if isinstance(f0_pred, torch.Tensor):
        f0_pred = f0_pred.cpu().numpy()
    if isinstance(f0_target, torch.Tensor):
        f0_target = f0_target.cpu().numpy()

    # Find voiced frames in both predicted and target
    voiced_pred = ~np.isnan(f0_pred) & (f0_pred > threshold)
    voiced_target = ~np.isnan(f0_target) & (f0_target > threshold)

    # Only consider frames that are voiced in both
    voiced_both = voiced_pred & voiced_target

    if np.sum(voiced_both) == 0:
        return 0.0, 0.0

    # Calculate F0 error for voiced frames
    f0_pred_voiced = f0_pred[voiced_both]
    f0_target_voiced = f0_target[voiced_both]

    # Calculate absolute error
    abs_error = np.abs(f0_pred_voiced - f0_target_voiced)

    # Calculate error rate (percentage of frames with error > threshold)
    error_threshold = 50  # Hz
    error_frames = np.sum(abs_error > error_threshold)
    error_rate = (error_frames / len(abs_error)) * 100

    # Calculate mean error
    mean_error = np.mean(abs_error)

    return error_rate, mean_error
