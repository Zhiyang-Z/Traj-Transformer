import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class VanillaAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B L (h d) -> B h L d", h=self.heads)
        k = rearrange(k, "B L (h d) -> B h L d", h=self.heads)
        v = rearrange(v, "B L (h d) -> B h L d", h=self.heads)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "B h L d -> B L (h d)")
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x