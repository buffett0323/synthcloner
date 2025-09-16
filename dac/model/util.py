import torch
import torch.nn.functional as F


# Repeat ADSR embedding for each onset position
def repeat_adsr_by_onset(proto_E: torch.Tensor,
                         onset_flags: torch.Tensor) -> torch.Tensor:
    """
    proto_E     : (B, C, Lp)
    onset_flags : (B, 1, T)  binary 0/1 or soft probs
    Returns
    --------
    adsr_stream : (B, C, T)
    """
    B, C, Lp = proto_E.shape
    T = onset_flags.size(-1)
    device = proto_E.device

    # find onset positions (B,N)
    # if soft flags, threshold >0.5 for indexing; you can also argrelmax
    onset_pos_list = [torch.where(onset_flags[b,0] > 0.5)[0] for b in range(B)]
    N_max = max(1, max(len(x) for x in onset_pos_list))  # ensure ≥1

    # build segment start index tensor (B,N_max)
    starts = torch.full((B, N_max), T, device=device, dtype=torch.long)
    for b, idx in enumerate(onset_pos_list):
        n = min(len(idx), N_max)
        starts[b, :n] = idx[:n]
        if n == 0:
            starts[b, 0] = 0  # dummy

    # segment ends (next onset or T)
    ends = torch.cat([starts[:,1:], torch.full_like(starts[:,:1], T)], dim=1)
    # lengths
    seg_len = (ends - starts).clamp(min=1)

    # frame indices [0..T-1]
    t = torch.arange(T, device=device)[None,None,:]              # (1,1,T)
    starts_ = starts[:,:,None]                                   # (B,N,1)
    ends_   = ends[:,:,None]

    # segment idx per frame: argmax over mask (broadcast)
    # mask True where t >= start & t < end
    seg_mask = (t >= starts_) & (t < ends_)
    # choose first True along N
    seg_idx = seg_mask.float().argmax(dim=1)                     # (B,T)

    # phase within segment
    seg_start_gather = starts.gather(1, seg_idx)                 # (B,T)
    phase = t.squeeze(0) - seg_start_gather                      # (B,T)
    # clip to proto length
    phase_clipped = phase.clamp(0, Lp-1)

    # gather proto
    idx = phase_clipped.unsqueeze(1).expand(-1,C,-1)             # (B,C,T)
    adsr_stream = torch.gather(proto_E, 2, idx)                  # (B,C,T)

    # zero out when phase >= Lp (segment longer than proto)
    mask_valid = (phase < Lp).float().unsqueeze(1)
    adsr_stream = adsr_stream * mask_valid
    return adsr_stream



# Gather note segments with padding
def gather_notes_pad(adsr_feat: torch.Tensor,
                     on_idx: list[list[int]],
                     L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract note segments from ADSR features and pad to fixed length.

    Args:
        adsr_feat: (B, 64, T) ADSR features
        on_idx: List of onset indices for each batch (sorted)
        L: Fixed length for padding

    Returns:
        note_E: (B, N_max, 64, L) Padded note segments
        mask: (B, N_max, L) Validity mask (1=valid, 0=padded)
    """
    B, C, T = adsr_feat.shape
    N_max = max(len(seq) for seq in on_idx)
    device = adsr_feat.device

    note_E = torch.zeros(B, N_max, C, L, device=device)
    mask   = torch.zeros(B, N_max, L, device=device)

    # Process each batch
    for b, starts in enumerate(on_idx):
        if len(starts) == 0:
            continue

        # Convert to tensors
        starts_tensor = torch.tensor(starts, device=device, dtype=torch.long)
        ends_tensor = torch.cat([
            starts_tensor[1:],
            torch.tensor([T], device=device, dtype=torch.long)
        ])

        # Calculate segment lengths
        seg_lengths = torch.minimum(ends_tensor - starts_tensor, torch.tensor(L, device=device))
        valid_mask = seg_lengths > 0

        if not valid_mask.any():
            continue

        # Get valid segments
        valid_starts = starts_tensor[valid_mask]
        valid_lengths = seg_lengths[valid_mask]
        valid_note_indices = torch.where(valid_mask)[0]

        # Assign segments to output
        for note_idx, start, length in zip(valid_note_indices, valid_starts, valid_lengths):
            end = start + length
            note_E[b, note_idx, :, :length] = adsr_feat[b, :, start:end]
            mask[b, note_idx, :length] = 1.0

    return note_E, mask


# Resample 64×T  →  note_E  B×N×64×L (Differentiable)
def resample_adsr(adsr_feat: torch.Tensor,
                  t_grid: torch.Tensor) -> torch.Tensor:
    B, C, T = adsr_feat.shape
    B2, N, L = t_grid.shape
    assert B == B2

    # Normalize grid coordinates to [-1, 1]
    grid_x = (t_grid / (T - 1) * 2 - 1).view(B * N, 1, L, 1)
    # Create 2D grid for grid_sample (second dimension is always 0)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.cat([grid_x, grid_y], dim=-1)  # (B*N, 1, L, 2)

    # Reshape input for grid_sample
    adsr4d = adsr_feat[:, :, None, :]          # (B,C,1,T)
    adsr4d = adsr4d.expand(-1, -1, N, -1).reshape(B * N, C, 1, T)

    sampled = F.grid_sample(adsr4d, grid,
                            mode='bilinear',
                            align_corners=True)      # (B*N,C,1,L)
    note_E = sampled.squeeze(2).view(B, N, C, L)
    return note_E                                    # (B,N,64,L)


# Apply ADSR envelope to onset sequence
def sequencer(proto_E: torch.Tensor,
              onset_flags: torch.Tensor) -> torch.Tensor:
    """
    Apply ADSR envelope prototype to onset sequence.

    Args:
        proto_E: (B, 64, L) ADSR envelope prototype
        onset_flags: (B, 1, T) Onset sequence flags

    Returns:
        (B, 64, T) ADSR stream applied to onset sequence
    """
    B, C, L = proto_E.shape
    _, _, T = onset_flags.shape

    # Reshape for grouped convolution
    weight = proto_E.view(B * C, 1, L)            # (B*C, 1, L)
    inp    = onset_flags.expand(-1, C, -1)        # (B,64,T)
    inp    = inp.reshape(1, B * C, T)             # (1, B*C, T)

    # Apply grouped convolution
    stream = F.conv1d(inp, weight,
                      groups=B * C,
                      padding=L - 1)              # (1, B*C, T+L-1)
    stream = stream[:, :, :T]                     # Trim to original length

    # Reshape back to batch and channel dimensions
    stream = stream.view(B, C, T)
    return stream


if __name__ == "__main__":

    # Your example onset flags
    onset = torch.tensor([
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    onset = onset.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 87)

    # Create dummy ADSR embedding
    tmp_len = 20
    adsr_embed = torch.zeros(1, 64, 87)
    for i in range(tmp_len):
        adsr_embed[:, :, i] = i * torch.ones(1, 64)

    # adsr_embed[:, :, 0] = 99 * torch.ones(1, 64)
    # adsr_embed[:, :, 1:tmp_len+1] = torch.randn(1, 64, tmp_len)

    # print(f"Onset shape: {onset.shape}")
    # print(f"ADSR embed shape: {adsr_embed.shape}")
    # print(f"Onset positions: {torch.where(onset[0, 0] == 1)[0].tolist()}")

    # Test the function
    result = repeat_adsr_by_onset(adsr_embed, onset)
    print(f"Result shape: {result.shape}")
    print("0:", result[:, :, 0])
    print("1:", result[:, :, 1])
    print("2:", result[:, :, 2])
    print("3:", result[:, :, 3])
    print("4:", result[:, :, 4])
    print("40:", result[:, :, 40])
    print(f"{tmp_len}:", result[:, :, tmp_len])
    # print(result[:, :, tmp_len+1])
    print("49:", result[:, :, 49])
    print("50:", result[:, :, 50])
    print("-1:", result[:, :, -1])


    # # Verify the concept: check that ADSR is applied correctly
    # onset_positions = torch.where(onset[0, 0] == 1)[0]
    # print(f"\nOnset positions: {onset_positions.tolist()}")

    # for i, start_pos in enumerate(onset_positions):
    #     if i + 1 < len(onset_positions):
    #         end_pos = onset_positions[i + 1]
    #     else:
    #         end_pos = onset.shape[-1]

    #     segment_length = end_pos - start_pos
    #     print(f"Segment {i}: position {start_pos} to {end_pos} (length: {segment_length})")

    #     # Check that the segment is not all zeros (ADSR was applied)
    #     segment_data = result[0, :, start_pos:end_pos]
    #     non_zero_count = (segment_data != 0).sum().item()
    #     print(f"  Non-zero elements in segment: {non_zero_count}/{segment_data.numel()}")

    # print(result[:,:,0])
    # print(result[:,:,1])
    # print(result[:,:,43])
    # print(result[:,:,44])
