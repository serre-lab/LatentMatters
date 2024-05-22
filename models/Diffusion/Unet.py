import torch.nn as nn
from .block import ConvNextBlock, ConvAttnBlock, Downsample, Upsample, ConvAttnBlock_SmallConv, ConvNextBlock_SmallConv
from .block import Residual, PreNorm, Attention, default
from .block import SinusoidalPositionEmbeddings
from functools import partial
import torch


class Unet_LDM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_feat=256,
                 embedding_model=None,
                 dim_mults=(1, 2, 4),
                 num_attn=1,
                 small_conv=False):
        super(Unet_LDM, self).__init__()

        self.in_channels = in_channels
        self.num_attn = num_attn
        init_dim = (n_feat // 3) * 2

        if embedding_model == 'stack':
            if small_conv:
                self.init_conv = nn.Conv2d(2 * self.in_channels, init_dim, 3, padding=1)
            else:
                self.init_conv = nn.Conv2d(2 * self.in_channels, init_dim, 7, padding=3)
        else:
            if small_conv:
                self.init_conv = nn.Conv2d(self.in_channels, init_dim, 3, padding=1)
            else:
                self.init_conv = nn.Conv2d(self.in_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: n_feat * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))


        time_dim = n_feat * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(n_feat),
            nn.Linear(n_feat, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.embedding_model = embedding_model

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        if small_conv:
            ConvBlock = ConvAttnBlock_SmallConv
            block_klass = partial(ConvNextBlock_SmallConv, mult=2)
        else:
            ConvBlock = ConvAttnBlock
            block_klass = partial(ConvNextBlock, mult=2)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [ConvBlock(dim_in, dim_out, time_emb_dim=time_dim, cond_emb_dim=None)]
                    + [ConvBlock(dim_out, dim_out, time_emb_dim=time_dim, cond_emb_dim=None) for _ in
                       range(self.num_attn - 1)]
                    + [Downsample(dim_out) if not is_last else nn.Identity()]
                    # [
                    #     block_klass(dim_in, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     block_klass(dim_out, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    #     block_klass(dim_out, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     block_klass(dim_out, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    #     block_klass(dim_out, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     block_klass(dim_out, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    #     Downsample(dim_out) if not is_last else nn.Identity(),
                    # ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, cond_emb_dim=None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, cond_emb_dim=None)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [ConvBlock(dim_out * 2, dim_out, time_emb_dim=time_dim, cond_emb_dim=None) for _ in
                     range(self.num_attn - 1)]
                    + [ConvBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, cond_emb_dim=None)]
                    + [Upsample(dim_in) if not is_last else nn.Identity()]
                    # [
                    #     block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     block_klass(dim_in, dim_in, time_emb_dim=time_dim, cond_emb_dim=None),
                    #     Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    #     Upsample(dim_in) if not is_last else nn.Identity(),
                    # ]
                )
            )
        out_dim = default(None, in_channels)

        self.final_conv = nn.Sequential(
            block_klass(n_feat, n_feat), nn.Conv2d(n_feat, out_dim, 1)
        )

    def forward(self, x, t, c=None, context_mask=None, verbose=False):
        if (c is not None) and (context_mask is not None):
            # mask out context if context_mask == 1
            context_mask = context_mask[:, None, None, None]
            context_mask = context_mask.repeat(1, x.size(1), x.size(2), x.size(3))
            context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
            c = c * context_mask

        if self.embedding_model == "stack":
            x = torch.cat([x, c], dim=1)
        x = self.init_conv(x)
        t = self.time_mlp(t)

        h = []

        # downsample

        # for block1,block2,attn1,block3,block4,attn2,downsample in self.downs:
        #     x = block1(x, t, c)
        #     x = block2(x, t, c)
        #     x = attn1(x)
        #     x = block3(x, t, c)
        #     x = block4(x, t, c)
        #     x = attn2(x)
        #     h.append(x)
        #     x = downsample(x)

        for idx, blocks in enumerate(self.downs):
            for block in blocks[:-1]:
                x = block(x, t, c)
                h.append(x)
            x = blocks[-1](x)
            if verbose:
                print(f'down block {idx+1} : {x.size()}')
        # bottleneck
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        if verbose:
            print(f'bottleneck : {x.size()}')
        x = self.mid_block2(x, t, c)

        # upsample
        # for block1, block2, attn, upsample in self.ups:
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = block1(x, t, c)
        #     x = block2(x, t, c)
        #     x = attn(x)
        #     x = upsample(x)

        for idx, blocks in enumerate(self.ups):
            for block in blocks[:-1]:
                x = torch.cat((x, h.pop()), dim=1)
                x = block(x, t, c)
            x = blocks[-1](x)
            if verbose:
                print(f'up block {idx + 1} : {x.size()}')
        return self.final_conv(x)