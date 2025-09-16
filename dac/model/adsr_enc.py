import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from .util import (
        gather_notes_pad,
        sequencer
    )
except ImportError:
    from util import (
        gather_notes_pad,
        sequencer
    )


# ADSR Encoder V1
class ResidualDilatedBlock(nn.Module):
    """
    Dilated 1-D residual block (kernel=3) with weight-norm Conv.
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation))
        self.act  = nn.GELU()

    def forward(self, x):
        return x + self.act(self.conv(x))


# ADSR Encoder V2, V3: Depthwise-separable convolution
class DSConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1):
        super().__init__()
        pad = (k - 1) // 2 * d
        self.depthwise = nn.Conv1d(in_ch, in_ch, k,
                                   groups=in_ch, dilation=d,
                                   padding=pad, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


# ADSR Encoder V2, V3: Basic convolution backbone
class ConvBackbone(nn.Module):
    """Convert log-RMS to amplitude features"""
    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1, dilation=2), nn.GELU(),
            nn.Conv1d(64, ch, 3, padding=1, dilation=4), nn.GELU(),
            nn.BatchNorm1d(ch)
        )

    def forward(self, x):
        return self.net(x)          # (B,64,T)


# ADSR Encoder V3: Depthwise-separable backbone
class DSBackbone(nn.Module):
    def __init__(self, pre_ch: int = 1, ch: int = 64):
        super().__init__()
        if ch == 64:
            self.net = nn.Sequential(
                nn.Conv1d(pre_ch, 32, 3, padding=1), nn.GELU(),          # early channel mix
                DSConv1d(32, 64, k=3, d=1),
                DSConv1d(64, 64, k=3, d=2),
                DSConv1d(64, 64, k=3, d=4),
                DSConv1d(64, 64, k=3, d=8),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(pre_ch,  64, 3, padding=1),  nn.GELU(),  # early mix
                DSConv1d( 64, 128, d=1),
                DSConv1d(128, 128, d=2),
                DSConv1d(128, 256, d=4),
                DSConv1d(256, 256, d=8),
            )

    def forward(self, x):
        return self.net(x)          # (B, 64, T)


# ADSR Encoder V2
class ResidualTCN(nn.Module):
    """
    Dilated residual temporal convolution block.

    Architecture:
    x --(depthwise dilated conv)--> BN/GELU --> 1×1 conv --+
      |                                                    |
      +---------------------------(residual add)-----------+

    Args:
        channels: Number of input/output channels
        k: Kernel size (usually 3)
        d: Dilation factor (2^i)
    """
    def __init__(self, channels: int, k: int = 3, d: int = 1):
        super().__init__()
        pad = (k - 1) * d                       # Maintain length

        # Depth-wise dilated conv (groups=channels)
        self.depthwise = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size=k,
                      dilation=d, padding=pad,
                      groups=channels, bias=False)
        )
        # Point-wise conv 1×1
        self.pointwise = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        )

        self.norm = nn.BatchNorm1d(channels)
        self.act  = nn.GELU()

    def forward(self, x):
        """
        x : (B, channels, T)
        """
        y = self.depthwise(x)
        y = self.pointwise(self.act(self.norm(y)))
        # Trim right side if padding causes length > T
        if y.size(-1) != x.size(-1):
            y = y[..., :x.size(-1)]
        return x + y


class ADSREncoderV1(nn.Module):
    """
    Temporal multi-scale ConvNet + BiLSTM.
    Input  : waveform  (B, 1, 44100)
    Output : ADSR embedding  (B, 64, 87)   # 44100 / 512 ≈ 86.1 ≈ 87
    Waveform (44100 samples)
        ↓
    Frame-based features (87 frames)
        ↓
    Multi-scale temporal modeling
        ↓
    Bidirectional sequence processing
        ↓
    ADSR embedding (64 dims × 87 frames)
    """
    def __init__(self,
                 hop: int = 512,
                 embed_channels: int = 64,
                 dilations=(1, 2, 4, 8, 16),
                 lstm_layers: int = 2,
                 lstm_hidden: int = 32):
        super().__init__()
        print(f"Initializing ADSR encoder V1 with embed_channels={embed_channels}")
        self.hop = hop                        # frame hop in samples
        self.eps = 1.0e-7

        # Envelope pre-processor
        # 1×1 conv (maps [log-RMS, Δlog-RMS] → C₀)
        self.pre = nn.Conv1d(2, embed_channels, kernel_size=1)

        # Dilated residual stack
        self.dilated = nn.ModuleList(
            [ResidualDilatedBlock(embed_channels, d) for d in dilations])

        # Low-rate context branch (4x downsample for global context)
        self.lowrate = nn.Sequential(
            nn.Conv1d(embed_channels, embed_channels, kernel_size=3,
                      padding=1, stride=4),
            nn.GELU())

        # BiLSTM over concatenated high+low streams
        self.bilstm = nn.LSTM(
            input_size=embed_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True)

        # 1×1 conv to final 64-d embedding
        self.out_proj = nn.Conv1d(lstm_hidden * 2, embed_channels, 1)

    def preprocess(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 1, T_frames)
        """
        length = wav.shape[-1]
        right_pad = math.ceil(length / self.hop) * self.hop - length
        wav = nn.functional.pad(wav, (0, right_pad))
        return wav


    # --------------------------------------------------------------------- #
    #  helper: frame-wise log-RMS + derivative
    # --------------------------------------------------------------------- #
    def _envelope_features(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 2, T_frames) : [log-RMS, Δlog-RMS]
        """
        # square + pool = frame energy
        rms = torch.sqrt(
            F.avg_pool1d(wav ** 2, kernel_size=self.hop, stride=self.hop) + self.eps
        )                                       # (B,1,Tf)

        log_rms = torch.log(rms + self.eps)          # (B,1,Tf)
        # first-order diff (pad with zero at t=0)
        diff = torch.cat([log_rms[:, :, :1],
                          log_rms[:, :, 1:] - log_rms[:, :, :-1]], dim=2)
        feats = torch.cat([log_rms, diff], dim=1)   # (B,2,Tf)
        return feats


    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # 0) preprocess
        wav = self.preprocess(wav)

        # 1) envelope features
        feats = self._envelope_features(wav)    # (B,2,Tf)

        # 2) initial projection
        x = self.pre(feats)                     # (B,C,Tf)

        # 3) dilated residual stack
        for block in self.dilated:
            x = block(x)                        # (B,C,Tf)

        # 4) Multi-scale concatenation (high-rate + up-sampled low-rate)
        low = self.lowrate(x)                   # (B,C,Tf/4)
        low = F.interpolate(low, size=x.shape[-1],
                            mode='linear', align_corners=False)
        x_cat = torch.cat([x, low], dim=1)      # (B,2C,Tf)

        # 5) BiLSTM (time dimension first for batch-first=True)
        x_cat = x_cat.transpose(1, 2)           # (B,Tf,2C)
        lstm_out, _ = self.bilstm(x_cat)        # (B,Tf,2*hidden)
        lstm_out = lstm_out.transpose(1, 2)     # (B,2*hidden,Tf)

        # 6) final projection → (B,64,Tf)
        z_a = self.out_proj(lstm_out)
        return z_a



# ADSR Encoder
class ADSREncoderV3(nn.Module):
    def __init__(self,
                 channels: int = 64,
                 hop: int = 512,
                 method: str = "length-weighted"):
        super().__init__()
        self.hop = hop
        self.eps = 1.0e-7

        self.backbone = DSBackbone(pre_ch=1, ch=channels)
        self.method = method

    def preprocess(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 1, T_frames)
        """
        length = wav.shape[-1]
        right_pad = math.ceil(length / self.hop) * self.hop - length
        wav = nn.functional.pad(wav, (0, right_pad))
        return wav

    def _envelope_features(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T_samples) in [-1,1]
        returns (B, 1, T_frames) : log-RMS
        """
        # square + pool = frame energy
        rms = torch.sqrt(
            F.avg_pool1d(wav ** 2, kernel_size=self.hop, stride=self.hop) + self.eps
        )                                       # (B,1,Tf)
        log_rms = torch.log(rms + self.eps)          # (B,1,Tf)
        return log_rms

    def forward(self,
                wav: torch.Tensor,          # (B,1,T_samples) waveform input
                onset_flags: torch.Tensor   # (B,1,T_frames)  0/1 impulse train
                ) -> dict:

        # 0) Preprocess waveform and calculate log-RMS
        wav = self.preprocess(wav)          # (B,1,T_samples)
        log_rms = self._envelope_features(wav)  # (B,1,T_frames)

        B, _, P = log_rms.shape

        # 1) Get on-set index list
        on_idx = [torch.where(onset_flags[b, 0] == 1)[0].tolist() for b in range(B)]

        # 2) DSConv backbone
        adsr_feat = self.backbone(log_rms)  # (B,64,T)

        # 3) Zero-pad per note  →    note_E, mask
        note_E, mask = gather_notes_pad(
            adsr_feat, on_idx, P)             # (B,N,64,L), (B,N,L)

        # 4) Length-weighted averaging
        if self.method == "length-weighted":
            # Mask-based averaging: normalize each note by valid length
            lengths = mask.sum(-1, keepdim=True)                    # (B,N,1)
            w = lengths / (lengths.sum(1, keepdim=True) + self.eps) # (B,N,1)
            w = w.unsqueeze(-1)                                     # (B,N,1,1)
            proto_E = (note_E * w).sum(dim=1)                       # (B,64,L)

        elif self.method == "equal-weighted":
            valid = mask.unsqueeze(2)                 # (B,N,1,L)
            sum_notes = (note_E * valid).sum(dim=1)   # (B,C,L)
            num_notes = valid.sum(dim=1).clamp(min=1) # (B,1,L)
            proto_E = sum_notes / num_notes

        else:
            raise ValueError(f"Invalid method: {self.method}")

        return proto_E


# V3 without folding
class ADSREncoderV4(nn.Module):
    """
    Input : waveform (B,1,T_samples)
    Output: ADSR feature map (B, 64, T_frames)
      – log-RMS and its derivative are extracted as in V1.
      – a very small DSConv stack does the temporal modelling.
    No onset / note pooling; keep the model light & fully frame-aligned.
    """
    def __init__(self,
                 channels: int = 64,
                 hop: int = 512):
        super().__init__()
        print(f"Initializing ADSR encoder V4 with channels={channels}")
        self.hop = hop
        self.eps = 1e-7
        self.backbone = DSBackbone(pre_ch=1, ch=channels)


    def _pad(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.size(-1)
        pad = (math.ceil(length / self.hop) * self.hop) - length
        return F.pad(wav, (0, pad))


    def _env_feats(self, wav: torch.Tensor) -> torch.Tensor:
        """Return (B,2,T_frames)"""
        rms = torch.sqrt(
            F.avg_pool1d(wav**2, kernel_size=self.hop, stride=self.hop) + self.eps
        )
        log_rms = torch.log(rms + self.eps)
        return log_rms


    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav : (B,1,T_samples)  range [-1,1]
        returns ADSR features (B,64,T_frames)
        """
        wav   = self._pad(wav)
        feats = self._env_feats(wav)          # (B,2,T)
        z_a   = self.backbone(feats)          # (B,64,T)
        return z_a



# ADSR encoder with content alignment
class ADSR_Content_Align(nn.Module):
    def __init__(self,
                 content_dim: int = 256,
                 adsr_dim: int = 64,
                 hidden_dim: int = 512,
                 num_heads: int = 8):
        super().__init__()

        # Project to common attention space
        self.content_query_proj = nn.Linear(content_dim, hidden_dim)
        self.adsr_key_proj = nn.Linear(adsr_dim, hidden_dim)
        self.adsr_value_proj = nn.Linear(adsr_dim, hidden_dim)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output projection back to content space
        self.output_proj = nn.Linear(hidden_dim, content_dim)
        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self,
                content_embedding: torch.Tensor,
                adsr_embedding: torch.Tensor
                ) -> torch.Tensor:
        """
        content_embedding: [B, 256, T]
        adsr_embedding: [B, 64, T]
        """
        content_embedding = content_embedding.transpose(1, 2) # [B, T, 256]
        adsr_embedding = adsr_embedding.transpose(1, 2) # [B, T, 64]

        # Project to attention space
        query = self.content_query_proj(content_embedding)    # [B, T, hidden_dim]
        key = self.adsr_key_proj(adsr_embedding)             # [B, T, hidden_dim]
        value = self.adsr_value_proj(adsr_embedding)         # [B, T, hidden_dim]

        # Cross-attention: content queries attend to ADSR keys/values
        attn_out, _ = self.cross_attention(
            query=query,    # [B, T, hidden_dim] - content embedding
            key=key,        # [B, T, hidden_dim] - ADSR embedding
            value=value     # [B, T, hidden_dim] - ADSR embedding
        )


        # Residual connection + norm
        output = self.norm(attn_out + query)

        # Project back to content dimension
        return self.output_proj(output).transpose(1, 2) # [B, T, 256] -> [B, 256, T]


# MLP
class ResidualFF(nn.Module):
    """Single position‑wise feed‑forward (1×1 Conv) block with residual & pre‑norm."""
    def __init__(self, dim: int, hidden: int,
                 dropout: float = 0.1, res_scale: float = 0.1):

        super().__init__()
        # Layer Norm makes the model not affected by original energy amplitude
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Conv1d(dim, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_scale = res_scale
        nn.init.zeros_(self.fc2.weight)  # Initialize close to identity mapping
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        y = self.ln(x.transpose(1, 2)).transpose(1, 2)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y) * self.res_scale
        return x + y # (B, D, T)


# MLP for content-adsr alignment
class ADSR_Content_MLP(nn.Module):
    """Stack of position‑wise residual FF blocks to refine (B,T,D) latent Z.

    Args:
        dim:     feature dimension (default 256)
        hidden:  hidden dimension of each FF block (default 512)
        n_layers:number of residual FF layers (default 5)
        dropout: dropout rate inside each block (default 0.1)
        res_scale: residual scale (<1 to stabilize deep stacks)
    """
    def __init__(self, dim: int = 256, hidden: int = 512,
                 n_layers: int = 10, dropout: float = 0.1, res_scale: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualFF(dim, hidden, dropout, res_scale)
                for _ in range(n_layers)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, D, T)
        for blk in self.layers:
            z = blk(z)
        return z


# FiLM Module (kept from original)
class ADSRFiLM(nn.Module):
    """
    FiLM-gates a content latent with an ADSR embedding.

        C_hat = γ ⊙ C + β
    where γ, β are learned linear projections of A.

    A : (B, adsr_ch,  T)
    C : (B, cont_ch,  T)
    ----------------------------------------------------------
    """
    def __init__(self,
                 adsr_ch:   int = 64,
                 cont_ch:   int = 256,
                 hidden_ch: int = 128):
        super().__init__()

        # Conv stack: (A) -> 2*cont_ch channels  (γ‖β)
        self.to_film = nn.Sequential(
            nn.Conv1d(adsr_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch, 2 * cont_ch, kernel_size=1)
        )

        # initialise to identity: γ=1, β=0
        with torch.no_grad():
            self.to_film[-1].weight.zero_()
            self.to_film[-1].bias.zero_()
            # Set γ to 1 for identity transformation
            self.to_film[-1].bias[:cont_ch] = 0.0  # β = 0
            self.to_film[-1].bias[cont_ch:] = 1.0  # γ = 1

    def forward(self, A: torch.Tensor, C: torch.Tensor):
        """
        A : (B, adsr_ch,  T)
        C : (B, cont_ch,  T)
        --------------------------------------------------
        returns
        C_tilde : (B, cont_ch, T)  -- FiLM-modulated content
        """
        γβ = self.to_film(A)                # (B, 2·C, T)
        γ, β = γβ.chunk(2, dim=1)           # split along channel
        C_tilde = γ * C + β                 # Standard FiLM formula: y = γ ⊙ x + β
        return C_tilde


if __name__ == "__main__":

    # # Your example onset flags
    # onset = torch.tensor([
    #     1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # onset = onset.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 87)

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    # model = ADSREncoderV3(channels=64).to(device)

    # p_onset = onset.to(device)
    # wav, _ = torchaudio.load("/mnt/gestalt/home/buffett/EDM_FAC_LOG/0716_mn/sample_audio/iter_0/conv_both/1_ref.wav")
    # wav = wav.unsqueeze(0)
    # wav = wav.to(device)

    # out = model(wav, p_onset)
    # print(out.shape) # 1, 64, 87

    B, D, T = 4, 256, 87
    z = torch.randn(B, D, T)
    model = ADSR_Content_MLP(dim=D, hidden=512, n_layers=10).to(device)
    z = z.to(device)
    out = model(z)
    print(out.shape)  # torch.Size([4, 256, 87])
