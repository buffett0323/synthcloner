# moco_transformer.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather  (無梯度).  若未啟用 torch.distributed 直接回傳原張量。
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo v2) for *Transformer-based timbre encoder*.
    - queue shape: (K, dim)
    - supports mlp=False  (MoCo v1)  or mlp=True (v2)
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        dim: int = 256,
        K: int = 512, # queue size
        m: float = 0.999,  # momentum for key encoder
        T: float = 0.2, # 0.07 for MoCo v1
        mlp: bool = True, # False for MoCo v1
    ):
        super().__init__()

        self.K, self.m, self.T = K, m, T

        # 1) build online & momentum encoders (deepcopy!)
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        # 2) projection heads ------------------------------------------------
        if mlp:
            self.head_q = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
            )
            self.head_k = copy.deepcopy(self.head_q)
            for p in self.head_k.parameters():
                p.requires_grad = False
        else:                          # MoCo v1
            self.head_q, self.head_k = nn.Identity(), nn.Identity()

        # 3) create the queue -----------------------------------------------
        self.register_buffer("queue", F.normalize(torch.randn(K, dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _momentum_update(self):
        """update key encoder & projector"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """keys: (B, dim)  L2-normalized"""
        keys = concat_all_gather(keys)                 # 跨 GPU 收集
        b = keys.size(0)
        ptr = int(self.queue_ptr)
        assert self.K % b == 0

        self.queue[ptr : ptr + b] = keys               # 直接覆寫
        ptr = (ptr + b) % self.K
        self.queue_ptr[0] = ptr

    # --------------------------------------------------------------------- #
    def forward(self, x_q, x_k):
        """
        x_q / x_k: audio (or latent) 張量，傳入 encoder
        return: contrastive loss (InfoNCE)
        """
        # --- online branch -------------------------------------------------
        q = F.normalize(self.head_q(self.encoder_q(x_q, None, None)), dim=1)  # (B,dim)

        # --- momentum branch ----------------------------------------------
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.head_k(self.encoder_k(x_k, None, None)), dim=1)

        # --- logits --------------------------------------------------------
        l_pos = torch.einsum("bd,bd->b", q, k).unsqueeze(1)       # (B,1)
        l_neg = torch.einsum("bd,kd->bk", q, self.queue.clone())  # (B,K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # --- update queue --------------------------------------------------
        self._dequeue_and_enqueue(k)

        # return loss (可選回傳 logits/labels 方便外部計算)
        # loss = F.cross_entropy(logits, labels)
        return logits, labels
