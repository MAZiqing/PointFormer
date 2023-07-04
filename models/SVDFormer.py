# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
from torch import nn, einsum
from einops import repeat, rearrange


# classes

class SVDTransformer(nn.Module):
    def __init__(
            self,
            *,
            in_dim,
            hid_dim,
            out_dim,
            # heads=4,
            dim_head=64,
            mode=32,
    ):
        super().__init__()
        # self.heads = heads
        self.mode = mode
        # inner_dim = dim_head * heads

        self.mlp_q = nn.Linear(in_dim, hid_dim)
        self.mlp_k = nn.Linear(in_dim, hid_dim)
        self.mlp_v = nn.Linear(in_dim, hid_dim)

        self.mlp_d = nn.Linear(2, 1)
        self.mlp_out = nn.Linear(hid_dim, out_dim)

        # self.num_neighbors = num_neighbors
        #
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_out = nn.Linear(inner_dim, dim)
        #
        # self.pos_mlp = nn.Sequential(
        #     nn.Linear(3, pos_mlp_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(pos_mlp_hidden_dim, inner_dim)
        # )
        #
        # attn_inner_dim = inner_dim * attn_mlp_hidden_mult
        #
        # self.attn_mlp = nn.Sequential(
        #     nn.Conv2d(inner_dim, attn_inner_dim, 1, groups=heads),
        #     nn.ReLU(),
        #     nn.Conv2d(attn_inner_dim, inner_dim, 1, groups=heads),
        # )

    def attn(self, q, k, v):
        B, L1, m1 = q.shape
        B, L2, m2 = k.shape
        sim = einsum('b i m, b i n -> b m n', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b m n, b i n -> b i m', attn, v)
        return out

    def forward(self, q, k, v, mask=None):
        B, T, H, W, C = q.shape
        B, K, H, W, C = q.shape
        q = self.mlp_q(q)
        k = self.mlp_q(k)
        v = self.mlp_q(v)

        B, T, H, W, D = q.shape
        q, k, v = map(lambda t: rearrange(t, 'b t h w c -> b (h w) (t c)'), (q, k, v))

        Uq, dq, Vq = torch.linalg.svd(q)
        Uk, dk, Vk = torch.linalg.svd(k)
        Uv, dv, Vv = torch.linalg.svd(v)

        mode = min(self.mode, H*W, min(T, K)*D)
        Uout = self.attn(Uq[..., :mode], Uk[..., :mode], Uv[..., :mode])
        assert Uout.shape == Uq[..., :mode].shape
        dout = self.mlp_d(torch.stack([dq[..., :mode], dk[..., :mode]], dim=-1)).squeeze(-1)
        assert dout.shape == dq[..., :mode].shape
        Vout = self.attn(Vq[..., :mode, :],
                         Vk[..., :mode, :],
                         Vv[..., :mode, :])
        assert Vout.shape == Vq[..., :mode, :].shape
        out = torch.einsum('b l m, b m -> b l m', Uout, dout)
        out = torch.einsum('b l m, b m n -> b l n', out, Vout)

        out = rearrange(out, 'b (h w) (t c) -> b t h w c', h=H, t=T)
        out = self.mlp_out(out)

        return out



        # n, h, num_neighbors = x.shape[1], self.heads, self.num_neighbors
        #
        # # get queries, keys, values
        #
        # q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        #
        # # split out heads
        #
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        #
        # # calculate relative positional embeddings
        #
        # rel_pos = rearrange(pos, 'b i c -> b i 1 c') - rearrange(pos, 'b j c -> b 1 j c')
        # rel_pos_emb = self.pos_mlp(rel_pos)
        #
        # # split out heads for rel pos emb
        #
        # rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h=h)
        #
        # # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        #
        # qk_rel = rearrange(q, 'b h i d -> b h i 1 d') - rearrange(k, 'b h j d -> b h 1 j d')
        #
        # # prepare mask
        #
        # if exists(mask):
        #     mask = rearrange(mask, 'b i -> b i 1') * rearrange(mask, 'b j -> b 1 j')
        #
        # # expand values
        #
        # v = repeat(v, 'b h j d -> b h i j d', i=n)
        #
        # # determine k nearest neighbors for each point, if specified
        #
        # if exists(num_neighbors) and num_neighbors < n:
        #     rel_dist = rel_pos.norm(dim=-1)
        #
        #     if exists(mask):
        #         mask_value = max_value(rel_dist)
        #         rel_dist.masked_fill_(~mask, mask_value)
        #
        #     dist, indices = rel_dist.topk(num_neighbors, largest=False)
        #
        #     indices_with_heads = repeat(indices, 'b i j -> b h i j', h=h)
        #
        #     v = batched_index_select(v, indices_with_heads, dim=3)
        #     qk_rel = batched_index_select(qk_rel, indices_with_heads, dim=3)
        #     rel_pos_emb = batched_index_select(rel_pos_emb, indices_with_heads, dim=3)
        #
        #     if exists(mask):
        #         mask = batched_index_select(mask, indices, dim=2)
        #
        # # add relative positional embeddings to value
        #
        # v = v + rel_pos_emb
        #
        # # use attention mlp, making sure to add relative positional embedding first
        #
        # attn_mlp_input = qk_rel + rel_pos_emb
        # attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')
        #
        # sim = self.attn_mlp(attn_mlp_input)
        #
        # # masking
        #
        # if exists(mask):
        #     mask_value = -max_value(sim)
        #     mask = rearrange(mask, 'b i j -> b 1 i j')
        #     sim.masked_fill_(~mask, mask_value)
        #
        # # attention
        #
        # attn = sim.softmax(dim=-2)
        #
        # # aggregate
        #
        # v = rearrange(v, 'b h i j d -> b i j (h d)')
        # agg = einsum('b d i j, b i j d -> b i d', attn, v)
        #
        # # combine heads
        #
        # return self.to_out(agg)


if __name__ == '__main__':
    model = SVDTransformer(in_dim=32, hid_dim=32, out_dim=32)
    q = torch.randn(1, 13, 8, 8, 32)
    k = torch.randn(1, 10, 8, 8, 32)
    out = model.forward(q, k, k)
    a = 1
