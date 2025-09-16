import math
from typing import List, Union, Optional

import numpy as np
import torch.nn.functional as F
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn, sin, pow

from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from dac.nn.quantize import ResidualVectorQuantize
from .encodec import SConv1d, SConvTranspose1d, SLSTM
from .transformer import TransformerEncoder, AttentionPooling
from .gradient_reversal import GradientReversal
from .adsr_enc import (
    ADSREncoderV1, ADSREncoderV3, ADSREncoderV4,
    ADSR_Content_Align, ADSRFiLM, ADSR_Content_MLP
)
from .util import repeat_adsr_by_onset
from alias_free_torch import Activation1d
from einops.layers.torch import Rearrange


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        conv1d_type = SConv1d# if causal else WNConv1d
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            conv1d_type(dim, dim, kernel_size=7, dilation=dilation, padding=pad, causal=causal, norm='weight_norm'),
            Snake1d(dim),
            conv1d_type(dim, dim, kernel_size=1, causal=causal, norm='weight_norm'),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class CNNLSTM(nn.Module):
    def __init__(self, indim, outdim, head, global_pred=False):
        super().__init__()
        self.global_pred = global_pred
        self.model = nn.Sequential(
            ResidualUnit(indim, dilation=1),
            ResidualUnit(indim, dilation=2),
            ResidualUnit(indim, dilation=3),
            Activation1d(activation=SnakeBeta(indim, alpha_logscale=True)),
            Rearrange("b c t -> b t c"),
        )
        self.heads = nn.ModuleList([nn.Linear(indim, outdim) for i in range(head)])

    def forward(self, x):
        # x: [B, C, T]
        x = self.model(x)
        if self.global_pred:
            x = torch.mean(x, dim=1, keepdim=False)
        outs = [head(x) for head in self.heads]
        return outs


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, causal: bool = False):
        super().__init__()
        conv1d_type = SConv1d# if causal else WNConv1d
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv1d_type(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                causal=causal,
                norm='weight_norm',
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        causal: bool = False,
        lstm: int = 2,
    ):
        super().__init__()
        conv1d_type = SConv1d# if causal else WNConv1d
        # Create first convolution
        self.block = [conv1d_type(1, d_model, kernel_size=7, padding=3, causal=causal, norm='weight_norm')]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, causal=causal)]

        # Add LSTM if needed
        self.use_lstm = lstm
        if lstm:
            self.block += [SLSTM(d_model, lstm)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            conv1d_type(d_model, d_latent, kernel_size=3, padding=1, causal=causal, norm='weight_norm'),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, causal: bool = False):
        super().__init__()
        conv1d_type = SConvTranspose1d #if causal else WNConvTranspose1d
        self.block = nn.Sequential(
            Snake1d(input_dim),
            conv1d_type(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                causal=causal,
                norm='weight_norm'
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        lstm: int = 2,
    ):
        super().__init__()
        conv1d_type = SConv1d# if causal else WNConv1d
        # Add first conv layer
        layers = [conv1d_type(input_channel, channels, kernel_size=7, padding=3, causal=causal, norm='weight_norm')]

        if lstm:
            layers += [SLSTM(channels, num_layers=lstm)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, causal=causal)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            conv1d_type(output_dim, d_out, kernel_size=7, padding=3, causal=causal, norm='weight_norm'),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)




class ABL_DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = 256,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        lstm: int = 2,
        causal: bool = False,
        timbre_classes: int = 428,
        pitch_nums: int = 88,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim, causal=causal, lstm=lstm)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Content
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        # Timbre
        self.transformer = TransformerEncoder(
            enc_emb_tokens=None,
            encoder_layer=4,
            encoder_hidden=256,
            encoder_head=4,
            conv_filter_size=1024,
            conv_kernel_size=5,
            encoder_dropout=0.1,
            use_cln=False,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            lstm=lstm,
            causal=causal,
        )

        self.apply(init_weights)

        # Predictors
        self.pitch_predictor = CNNLSTM(latent_dim, pitch_nums, head=1, global_pred=False)
        self.timbre_predictor = CNNLSTM(latent_dim, timbre_classes, head=1, global_pred=True)


        # Conditional LayerNorm
        self.style_linear = nn.Linear(latent_dim, latent_dim * 2)
        with torch.no_grad():
            nn.init.zeros_(self.style_linear.bias)
            self.style_linear.bias.data[:latent_dim] = 1
            self.style_linear.bias.data[latent_dim:] = 0
        self.style_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)


    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data


    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: Optional[int] = None,
    ):
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss


    def decode(self, z: torch.Tensor):
        return self.decoder(z)


    def forward(
        self,
        audio_data: torch.Tensor,
        content_match: torch.Tensor = None,
        timbre_match: torch.Tensor = None,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):

        length = audio_data.shape[-1]

        # 0. Preprocess
        content_match = self.preprocess(content_match, sample_rate)
        timbre_match = self.preprocess(timbre_match, sample_rate)

        # Perturbation's encoders
        content_match_z = self.encoder(content_match)
        timbre_match_z = self.encoder(timbre_match)

        # 1. Disentangle Content
        cont_z, _, _, cont_commitment_loss, cont_codebook_loss = self.quantizer(
            content_match_z, n_quantizers
        )

        # 3. Disentangle Timbre
        timbre_match_z = timbre_match_z.transpose(1, 2) # (B, D, T)
        timbre_match_z = self.transformer(timbre_match_z, None, None) # (B, T', D)
        timbre_match_z = timbre_match_z.transpose(1, 2) # (B, D, T')
        timbre_match_z = torch.mean(timbre_match_z, dim=2) # (B, D)


        # 6-1. Predictors
        # timbre_match_z = F.normalize(timbre_match_z, dim=-1)
        pred_pitch     = self.pitch_predictor(cont_z)[0]
        pred_timbre_id = self.timbre_predictor(timbre_match_z.unsqueeze(-1))[0]


        # Project timbre latent to style parameters
        style = self.style_linear(timbre_match_z).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)


        # 7. Conditional Layer Norm
        z_cln = cont_z.transpose(1, 2) # (B, T, D)
        z_cln = self.style_norm(z_cln) # (B, T, D)
        z_cln = z_cln.transpose(1, 2) # (B, D, T)
        z_cln = z_cln * gamma + beta

        # 8. Decoder
        x = self.decode(z_cln)

        return {
            "audio": x[..., :length],
            # "z_gt": z_gt,
            # "codes": cont_codes,
            # "latents": cont_latents,
            "vq/commitment_loss": cont_commitment_loss,
            "vq/codebook_loss": cont_codebook_loss,

            "pred_timbre_id": pred_timbre_id,
            "pred_pitch": pred_pitch,
        }


    @torch.no_grad()
    def conversion(
        self,
        orig_audio: torch.Tensor,
        ref_audio: torch.Tensor = None,
        sample_rate: int = None,
        n_quantizers: int = None,
        convert_type: str = "timbre",
    ):
        length = orig_audio.shape[-1]

        # 0. Preprocess
        orig_audio = self.preprocess(orig_audio, sample_rate)
        if ref_audio is not None:
            ref_audio = self.preprocess(ref_audio, sample_rate)

        # Perturbation's encoders
        orig_audio_z = self.encoder(orig_audio)
        if ref_audio is not None:
            ref_audio_z = self.encoder(ref_audio)

        # 1. Disentangle Content
        cont_z, _, _, cont_commitment_loss, cont_codebook_loss = self.quantizer(
            orig_audio_z, n_quantizers
        )

        # 3. Disentangle Timbre
        if convert_type in ["timbre"]:
            timbre_match_z = ref_audio_z
        else:
            timbre_match_z = orig_audio_z

        timbre_match_z = timbre_match_z.transpose(1, 2) # (B, D, T)
        timbre_match_z = self.transformer(timbre_match_z, None, None) # (B, T', D)
        timbre_match_z = timbre_match_z.transpose(1, 2) # (B, D, T')
        timbre_match_z = torch.mean(timbre_match_z, dim=2) # (B, D)



        # Predictors
        # timbre_match_z = F.normalize(timbre_match_z, dim=-1)
        pred_pitch     = self.pitch_predictor(cont_z)[0]
        pred_timbre_id = self.timbre_predictor(timbre_match_z.unsqueeze(-1))[0]

        # Project timbre latent to style parameters
        style = self.style_linear(timbre_match_z).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)

        # 6. Gradient Reversal
        # No gradient reversal for conversion

        # 7. Conditional Layer Norm
        z_cln = cont_z.transpose(1, 2) # (B, T, D)
        z_cln = self.style_norm(z_cln) # (B, T, D)
        z_cln = z_cln.transpose(1, 2) # (B, D, T)
        z_cln = z_cln * gamma + beta

        x = self.decode(z_cln)

        return {
            "audio": x[..., :length],
            # "codes": codes,
            # "latents": latents,
            "vq/commitment_loss": cont_commitment_loss,
            "vq/codebook_loss": cont_codebook_loss,

            "pred_pitch": pred_pitch,
            "pred_timbre_id": pred_timbre_id,
        }
