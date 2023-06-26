import torch
from einops import repeat
from torch import nn, einsum

from pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class PointTransformerLayer(nn.Module):
    def __init__(self, dim: int, emb_hidden: int, attn_expansion: int, 
                 nneighbors: int = None) -> None:
        super(PointTransformerLayer, self).__init__()
        self.w_qs = nn.Linear(dim, dim, bias = False)
        self.w_ks = nn.Linear(dim, dim, bias = False)
        self.w_vs = nn.Linear(dim, dim, bias = False)

        self.pos_emb = nn.Sequential(
            nn.Linear(3, emb_hidden),
            nn.ReLU(),
            nn.Linear(emb_hidden, dim),
        )

        self.attn = nn.Sequential(
            nn.Linear(dim, dim * attn_expansion),
            nn.ReLU(),
            nn.Linear(dim * attn_expansion, dim),
        )
        
        self.num_neighbors = nneighbors

    @staticmethod
    def batched_idx_select(values, indices, dim: int = 1):
        value_dims = values.shape[(dim + 1):]
        _, indices_shape = map(lambda t: list(t.shape), (values, indices))
        indices = indices[(..., *((None,) * len(value_dims)))]
        indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
        value_expand_len = len(indices_shape) - (dim + 1)
        values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

        value_expand_shape = [-1] * len(values.shape)
        expand_slice = slice(dim, (dim + value_expand_len))
        value_expand_shape[expand_slice] = indices.shape[expand_slice]
        values = values.expand(*value_expand_shape)

        return values.gather(dim + value_expand_len, indices)

    def forward(self, x, pos, mask = None):
        n, num_neighbors = x.shape[1], self.num_neighbors

        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        qk_rel = q[:, :, None, :] - k[:, None, :, :]
        v = repeat(v, 'b j d -> b i j d', i = n)

        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_emb(rel_pos)

        if mask is not None:
            mask = mask[:, :, None] * mask[:, None, :]

        # determine k nearest neighbors for each point, if specified
        if num_neighbors and num_neighbors < n:
            rel_dist = rel_pos.norm(dim = -1)

            if mask is not None:
                mask_value = torch.finfo(rel_dist.dtype).max
                rel_dist.masked_fill_(~mask, mask_value)

            _, indices = rel_dist.topk(num_neighbors, largest = False)

            v = self.batched_idx_select(v, indices, dim = 2)
            qk_rel = self.batched_idx_select(qk_rel, indices, dim = 2)
            rel_pos_emb = self.batched_idx_select(rel_pos_emb, indices, dim = 2)
            mask = self.batched_idx_select(mask, indices, dim = 2) if mask else None

        v = v + rel_pos_emb
        sim = self.attn(qk_rel + rel_pos_emb)

        if mask is not None:
            mask_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask[..., None], mask_value)

        attn = sim.softmax(dim = -2)
        agg = einsum('b i j d, b i j d -> b i d', attn, v)

        return agg

class TransformerBlock(nn.Module):
    def __init__(self, fdim: int, tdim: int, emb_hidden: int, 
                 attn_expansion: int, k: int = None) -> None:
        super(TransformerBlock, self).__init__()
        self.fc1 = nn.Linear(fdim, tdim)
        self.fc2 = nn.Linear(tdim, fdim)
        self.transformer = PointTransformerLayer(tdim, emb_hidden, attn_expansion, k)

    def forward(self, x, pos, mask = None):
        shortcut = x
        x = self.fc1(x)
        x = self.transformer(x, pos, mask)
        x = self.fc2(x) + shortcut

        return x

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels) -> None:
        super(TransitionDown, self).__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)

class SwapAxes(nn.Module):
    def __init__(self) -> None:
        super(SwapAxes, self).__init__()
    
    def forward(self, x):
        return x.transpose(1, 2)

class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out) -> None:
        super(TransitionUp, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
