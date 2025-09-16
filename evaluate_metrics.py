"""
Simplified metrics for evaluating the model.

- Multi-scale STFT Loss
- F0 Frame Error Rate (using torchcrepe like eval_f0_test.py)
- LogRMS Envelope Loss (L1)
"""

import torch
import torchaudio
import torchcrepe
import torch.nn.functional as F
from torch import nn
from audiotools import AudioSignal, STFTParams
import typing
from typing import List

# 1. MSTFT Loss
class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


# 2. LogRMS Envelope Loss (L1)
class LogRMSEnvelopeLoss(nn.Module):
    """Computes the L1 loss between log RMS envelopes of audio signals.

    This loss measures the difference between the RMS energy envelopes
    of predicted and target audio signals in the log domain.

    Parameters
    ----------
    frame_length : int, optional
        Length of each frame for RMS calculation, by default 2048
    hop_length : int, optional
        Number of samples between consecutive frames, by default 512
    weight : float, optional
        Weight of this loss, by default 1.0
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        weight: float = 1.0,
    ):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.weight = weight

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes L1 loss between log RMS envelopes.

        Parameters
        ----------
        x : AudioSignal
            Predicted signal
        y : AudioSignal
            Target signal

        Returns
        -------
        torch.Tensor
            LogRMS envelope L1 loss
        """
        # Extract audio tensors
        x_audio = x.audio_data.squeeze()  # Remove batch dimension if present
        y_audio = y.audio_data.squeeze()

        # Ensure we have 1D tensors
        if x_audio.dim() > 1:
            x_audio = x_audio.mean(dim=0)  # Average across channels if stereo
        if y_audio.dim() > 1:
            y_audio = y_audio.mean(dim=0)

        # Pad signals to ensure they have the same length
        max_length = max(x_audio.shape[-1], y_audio.shape[-1])
        x_audio = F.pad(x_audio, (0, max_length - x_audio.shape[-1]))
        y_audio = F.pad(y_audio, (0, max_length - y_audio.shape[-1]))

        # Compute RMS envelopes
        x_rms = self._compute_rms_envelope(x_audio)
        y_rms = self._compute_rms_envelope(y_audio)

        # Apply log transformation with small epsilon to avoid log(0)
        eps = 1e-8
        x_log_rms = torch.log(x_rms + eps)
        y_log_rms = torch.log(y_rms + eps)

        # Compute L1 loss
        loss = F.l1_loss(x_log_rms, y_log_rms)

        return self.weight * loss

    def _compute_rms_envelope(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute RMS envelope of audio signal using sliding windows.

        Parameters
        ----------
        audio : torch.Tensor
            Input audio signal (1D tensor)

        Returns
        -------
        torch.Tensor
            RMS envelope values
        """
        # Use unfold to create overlapping frames
        frames = audio.unfold(0, self.frame_length, self.hop_length)

        # Compute RMS for each frame
        rms_values = torch.sqrt(torch.mean(frames ** 2, dim=1))

        return rms_values


# 3. F0 Evaluation Loss (simplified using torchcrepe approach from eval_f0_test.py)
class F0EvalLoss(nn.Module):
    """Computes F0 correlation and RMSE metrics between reference and estimated audio signals.

    This loss measures the pitch accuracy between predicted and target audio signals
    using the CREPE model for F0 extraction and computes correlation and RMSE metrics.

    Parameters
    ----------
    hop_length : int, optional
        Number of samples between consecutive F0 frames, by default 160
    fmin : float, optional
        Minimum frequency for F0 detection, by default 50.0 Hz
    fmax : float, optional
        Maximum frequency for F0 detection, by default 1100.0 Hz
    model_size : str, optional
        CREPE model size ('tiny' or 'full'), by default 'tiny'
    voicing_thresh : float, optional
        Confidence threshold for voicing detection, by default 0.5
    weight : float, optional
        Weight of this loss, by default 1.0
    """

    def __init__(
        self,
        hop_length=160,
        fmin=50.0,
        fmax=1100.0,
        model_size='tiny',
        voicing_thresh=0.5,
        weight=1.0,
        device=torch.device('cpu'),
        sr_in=44100,
        sr_out=16000,
        resample_to_16k=True,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.model_size = model_size
        self.voicing_thresh = voicing_thresh
        self.weight = weight
        self.sr_in = sr_in
        self.sr_out = sr_out
        self.resampler = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out).to(device)
        self.resample_to_16k = resample_to_16k

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes F0 correlation and RMSE between predicted and target signals.

        Parameters
        ----------
        x : AudioSignal
            Predicted signal
        y : AudioSignal
            Target signal

        Returns
        -------
        torch.Tensor
            F0 evaluation loss (negative correlation + RMSE penalty)
        """

        # Extract audio tensors and ensure they're on the same device
        device = x.audio_data.device
        x_audio = x.audio_data.squeeze()  # Remove batch dimension if present
        y_audio = y.audio_data.squeeze()

        # Pad signals to ensure they have the same length
        max_length = max(x_audio.shape[-1], y_audio.shape[-1])
        x_audio = F.pad(x_audio, (0, max_length - x_audio.shape[-1]))
        y_audio = F.pad(y_audio, (0, max_length - y_audio.shape[-1]))

        # Add batch dimension for processing
        x_batch = x_audio.unsqueeze(0)  # (1, T)
        y_batch = y_audio.unsqueeze(0)  # (1, T)

        # Compute F0 metrics using the simplified approach from eval_f0_test.py
        f0_corr, f0_rmse = self._compute_f0_metrics_simple(x_batch, y_batch, device)

        # Convert to tensors and handle NaN values
        if torch.isnan(f0_corr) or torch.isnan(f0_rmse):
            # Return zero loss if metrics are invalid
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute loss: negative correlation (higher correlation = lower loss) + RMSE penalty
        # Normalize RMSE to reasonable range (typical RMSE is 0-100 cents)
        normalized_rmse = f0_rmse / 100.0

        # Loss = -correlation + RMSE penalty
        # Higher correlation reduces loss, higher RMSE increases loss
        loss = -f0_corr + normalized_rmse

        return self.weight * loss

    def _compute_f0_metrics_simple(self, ref_wavs: torch.Tensor, est_wavs: torch.Tensor, device: torch.device):
        """Compute F0 correlation and RMSE metrics using the simplified approach from eval_f0_test.py."""
        assert ref_wavs.shape == est_wavs.shape, "ref/est shape must match (B,T)"
        B, T = ref_wavs.shape

        # Resample to 16kHz if needed
        if self.resample_to_16k:
            ref_resampled = self.resampler(ref_wavs)  # (B, T')
            est_resampled = self.resampler(est_wavs)  # (B, T')
        else:
            ref_resampled = ref_wavs
            est_resampled = est_wavs

        # Extract F0 using torchcrepe (like in eval_f0_test.py)
        def crepe_f0(wav_1x1T):
            f0, pd = torchcrepe.predict(
                wav_1x1T, sample_rate=self.sr_out, hop_length=self.hop_length,
                fmin=self.fmin, fmax=self.fmax, model=self.model_size,
                device=device, pad=True, return_periodicity=True
            )  # (1,F)
            pd = torchcrepe.filter.median(pd, win_length=3)
            return f0, pd

        # Process each sample
        corr_vals, rmse_vals = [], []

        for i in range(B):
            # Extract F0 for reference and estimated
            f0r, pdr = crepe_f0(ref_resampled[i:i+1])
            f0e, pde = crepe_f0(est_resampled[i:i+1])

            # Apply voicing mask
            vr = pdr >= self.voicing_thresh
            ve = pde >= self.voicing_thresh
            joint = vr & ve

            # Convert to cents and compute metrics
            def hz_to_cents(hz):
                return 1200.0 * torch.log2(hz)

            valid = joint & torch.isfinite(f0r) & torch.isfinite(f0e) & (f0r > 0) & (f0e > 0)

            if valid.any():
                r = f0r[valid]
                e = f0e[valid]

                # RMSE in cents
                rmse_c = torch.sqrt(torch.mean((hz_to_cents(e) - hz_to_cents(r))**2))

                # Pearson corr in Hz
                r0 = r - r.mean()
                e0 = e - e.mean()
                corr = (r0 @ e0) / (torch.sqrt((r0**2).sum()*(e0**2).sum()) + 1e-12)

                if rmse_c <= 100:
                    corr_vals.append(corr)
                    rmse_vals.append(rmse_c)

        return corr_vals, rmse_vals
        # # Return mean values
        # mean_corr = torch.stack(corr_vals).mean()
        # mean_rmse = torch.stack(rmse_vals).mean()

        # return mean_corr, mean_rmse, high_rmse_paths, nan_paths

    def get_metrics(self, x: AudioSignal, y: AudioSignal):
        """Get detailed F0 metrics without computing loss.

        Returns
        -------
        dict
            Dictionary containing F0 correlation, RMSE, and other metrics
        """
        device = x.audio_data.device
        x_audio = x.audio_data.squeeze()
        y_audio = y.audio_data.squeeze()

        # Compute metrics
        f0_corr, f0_rmse = self._compute_f0_metrics_simple(x_audio, y_audio, device)

        return {
            "f0_corr": f0_corr, #.item(),
            "f0_rmse": f0_rmse, #.item(),
        }


if __name__ == "__main__":
    # Initialize the loss function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f0_loss = F0EvalLoss(
        hop_length=160,
        fmin=50.0,
        fmax=1100.0,
        model_size='tiny',
        voicing_thresh=0.5,
        weight=1.0,
        device=device
    ).to(device)
