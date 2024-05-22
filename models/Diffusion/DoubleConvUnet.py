import torch.nn as nn
import torch.nn.functional as F
import torch


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Double_cond(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Double_cond_skip(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_Double(nn.Module):
    def __init__(self, c_in=3, c_out=3, n_feat=64, time_dim=256, dim_mults=[1,2], remove_deep_conv=False, conditioning="stack"):
        super().__init__()
        self.n_feat = n_feat
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.conditioning = conditioning
        if self.conditioning == 'stack':
            c_in = 2*c_in

        self.inc = nn.Sequential(
            DoubleConv(c_in, n_feat),
            DoubleConv(n_feat, n_feat))

        dims = [*map(lambda m: n_feat * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.ModuleList(
                    [Down(dim_in, dim_out)] +
                    [SelfAttention(dim_out)]
            )
            )
        #self.downs.append(Double_cond(dim_out[-1], dim_out[-1]))

        #self.down1 = Down(n_feat, 2*n_feat)
        #self.sa1 = SelfAttention(2*n_feat)
        #self.down2 = Down(2*n_feat, 4*n_feat)
        #self.sa2 = SelfAttention(4*n_feat)
        #self.down3 = Down(256, 256)
        #self.down3 = Double_cond(4*n_feat, 4*n_feat)
        #self.sa3 = SelfAttention(4*n_feat)

        if remove_deep_conv:
            self.bot1 = DoubleConv(dims[-1], dims[-1])
            self.bot3 = DoubleConv(dims[-1], dims[-1])
        else:
            self.bot1 = DoubleConv(dims[-1], dims[-1]*2)
            self.bot2 = DoubleConv(dims[-1]*2, dims[-1]*2)
            self.bot3 = DoubleConv(dims[-1]*2, dims[-1])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(
                nn.ModuleList(
                    [Up(dim_in+dim_out, dim_in)] +
                    [SelfAttention(dim_in)]
                ))
            print(f'Up({dim_in+dim_out},{dim_in})')

        #self.up1 = Up(512, 128)
        #self.up1 = Double_cond_skip(8*n_feat, 4*n_feat)
        #self.sa4 = SelfAttention(4*n_feat)
        #self.up2 = Up(4*n_feat + 2*n_feat, 4*n_feat)
        #self.sa5 = SelfAttention(4*n_feat)
        #self.up3 = Up(4*n_feat + n_feat, 2*n_feat)
        #self.sa6 = SelfAttention(2*n_feat)

        self.outc = nn.Sequential(
                    DoubleConv(dims[0], dims[0]),
                    DoubleConv(dims[0], dims[0]),
                    nn.Conv2d(dims[0], c_out, kernel_size=1)
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t, c=None, context_mask=None, verbose=False):
        if (c is not None) and (self.conditioning == 'stack'):
            x = torch.cat([x, c], dim=1)
        h = []
        x = self.inc(x) # 128, 12, 12
        if verbose:
            print(f'init conv: {x.size()}')
        h.append(x)
        for idx, blocks in enumerate(self.downs):
            x = blocks[0](x, t)
            x = blocks[1](x)
            if verbose:
                print(f'Down {idx+1}: {x.size()}')
            if idx < len(self.downs)-1:
                h.append(x) #0:(256,6,6); 1:(512,3,3)

        x = self.bot1(x)
        if not self.remove_deep_conv:
            x = self.bot2(x)
            if verbose:
                print(f'Bneck: {x.size()}')
        x = self.bot3(x) #512,3,3

        for idx, blocks in enumerate(self.ups):
            x = blocks[0](x, h.pop(), t)
            x = blocks[1](x)
            if verbose:
                print(f'Down {idx+1}: {x.size()}')

        output = self.outc(x)
        return output

    def forward(self, x, t, c=None, context_mask=None, verbose=False):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t, c=c, context_mask=context_mask, verbose=verbose)