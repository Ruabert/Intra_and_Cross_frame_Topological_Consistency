import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np




def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def relative_pos_dis(height=32, weight=32, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    #dis = torch.exp(-dis*(1/(2*sita**2)))
    return  dis

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train"):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots/1.0 # T
        attn = self.attend(dots)
        #attentionheatmap_visual(attn, out_dir='./Visualization/ACDC/SETR')
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer_record(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            min_ax = torch.min(ax)
            max_ax = torch.max(ax)
            min_x = torch.min(x)
            max_x = torch.max(x)
            print(min_ax.item(), min_x.item(), max_ax.item(), max_x.item())
            x = ax + x
            x = ff(x) + x
            ftokens.append(x)
            attmaps.append(amap)
        return x, ftokens, attmaps


class Setr(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=8, dim=512, depth=12, heads=8, mlp_dim=512*4, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num
        patch_dim = n_channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # encoder
        x = self.transformer(x)  # b h*w ppc
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x

    def infere(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        xin = self.dropout(x)
        # encoder
        x, ftokens, attmaps = self.transformer.infere(xin)  # b h*w ppc
        ftokens.insert(0, xin)
        x = self.from_patch_embedding(x)  # b c h*w
        x = x.view(b, self.dmodel, self.image_height//self.patch_height, self.image_width//self.patch_width)
        # decoder
        x = self.decoder(x)
        return x, ftokens, attmaps










