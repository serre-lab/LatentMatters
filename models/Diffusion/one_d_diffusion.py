from torch import nn, einsum
import math
from einops import rearrange
import torch
from functools import partial
import torch.nn.functional as F


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MLP_time(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim):
        super().__init__()
        self.time_emb = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim_in))
            if exists(time_emb_dim)
            else None
        )
        self.mlp_block = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU()
        )

    def forward(self, x, t=None):
        h = x

        if exists(self.time_emb) and exists(t):
            assert exists(t), "time embedding must be passed in"
            condition = self.time_emb(t)
            h = h + condition

        h = self.mlp_block(h)
        return h

class One_d_UNet(nn.Module):
    def __init__(self, latent_size, features=[64, 32, 16]):
        super().__init__()
        self.latent_size = latent_size
        first_feature = features[0]
        self.features_list = [latent_size] + features
        in_out = list(zip(self.features_list[:-1], self.features_list[1:]))
        time_dim = 4*first_feature

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(first_feature),
            nn.Linear(first_feature, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(MLP_time(dim_in, dim_out, time_emb_dim=time_dim))

        mid_dim = features[-1]
        self.mid = MLP_time(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(MLP_time(dim_out*2, dim_in, time_emb_dim=time_dim))


    def forward(self, x, t):
        t_embed = self.time_embed(t)
        h = []
        for idx, blocks in enumerate(self.downs):
            x = blocks(x, t_embed)
            #print(f"down {idx} : {x.size()}")
            h.append(x)

        x = self.mid(x, t_embed)

        for idx, blocks in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = blocks(x, t_embed)

        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

def DownsampleMLP(dim, dim_out = None):
    return nn.Linear(dim, default(dim_out, dim))

def UpsampleMLP(dim, dim_out = None):
    return nn.Linear(dim, default(dim_out, dim))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class RMSNorm_MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class PreNorm_MLP(nn.Module):
    def __init__(self, dim, fn, norm_type='rms'):
        super().__init__()
        self.fn = fn
        if norm_type == "rms":
            self.norm = RMSNorm_MLP(dim)
        elif norm_type == "groupnorm":
            self.norm = GroupNorm32(32, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        if half_dim > 1:
            emb = math.log(self.theta) / (half_dim - 1)
        else:
            emb = math.log(self.theta) / (half_dim)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class Block_MLP(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock_MLP(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block_MLP(dim, dim_out, groups = groups)
        self.block2 = Block_MLP(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            #time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention_MLP(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        #self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            RMSNorm_MLP(dim)
        )

    def forward(self, x):
        #b, c, n = x.shape
        x = x.unsqueeze(-1)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        out = out.squeeze(-1)
        return self.to_out(out)

class MyAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        #self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            RMSNorm_MLP(dim)
        )

    def forward(self, x):
        #b, c, n = x.shape
        #x = x.unsqueeze(-1)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) -> b h c 1', h = self.heads), qkv)

        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k)*self.scale
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h c n -> b (h c) n')
        out = out.squeeze(-1)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention_MLP(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        #self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        #x = x.unsqueeze(-2)
        #b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) -> b h c', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h i, b h d j -> b h i j', q, k)
        #sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        latent_size=784,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        conditioning=None,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        norm_type='rms'
    ):
        super().__init__()

        # determine dimensions
        self.latent_size = latent_size
        self.channels = channels
        self.conditioning = conditioning
        input_channels = channels * (2 if self.conditioning == 'stack' else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, proto = None):

        x = x.view(-1, 1, self.latent_size)
        if self.conditioning == "stack" and (proto is not None):
            #x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            proto = proto.view(-1, 1, self.latent_size)
            x = torch.cat((proto, x), dim = 1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        return x[:,0,:]


class Unet1D_MLP(nn.Module):
    def __init__(
        self,
        dim,
        latent_size=784,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        conditioning=None,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        norm_type='rms',
        ae_enc = None
    ):
        super().__init__()

        # determine dimensions
        self.latent_size = latent_size
        self.channels = channels
        self.conditioning = conditioning
        #input_channels = channels * (2 if self.conditioning == 'stack' else 1)
        input_channels = latent_size * (2 if self.conditioning == 'stack' else 1)
        init_dim = default(init_dim, dim)
        #self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)
        self.init_conv = nn.Linear(input_channels, init_dim)
        self.ae_enc = ae_enc
        dims = [init_dim, *map(lambda m: int(dim // m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock_MLP, groups = resnet_block_groups)

        # time embeddings

        #time_dim = dim * 4
        time_dim = 128

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            #sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            sinu_pos_emb = SinusoidalPosEmb(128, theta=sinusoidal_pos_emb_theta)
            #fourier_dim = dim
            fourier_dim = 128

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                #Residual(PreNorm_MLP(dim_in, LinearAttention_MLP(dim_in))),
                Residual(PreNorm_MLP(dim_in, MyAttention(dim_in), norm_type=norm_type)),
                DownsampleMLP(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm_MLP(mid_dim, MyAttention(mid_dim, dim_head = attn_dim_head, heads = attn_heads), norm_type=norm_type))
        #self.mid_attn = Residual(
        #    PreNorm_MLP(mid_dim, LinearAttention_MLP(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm_MLP(dim_out, MyAttention(dim_out), norm_type=norm_type)),
                UpsampleMLP(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in)
            ]))

        default_out_dim = latent_size * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Linear(dim, self.out_dim)
        a=1

    def forward(self, x, time, proto = None, guidance_mask=None):
        if self.ae_enc is not None:
            x = self.ae_enc.get_latent_from_img(x, mean=0, std=1, scale=False)
            if proto is not None:
                proto = self.ae_enc.get_latent_from_img(proto, mean=0, std=1, scale=False).detach()
        #x = x.view(-1, 1, self.latent_size)
        #x = x.view(-1, self.latent_size)
        if (proto is not None) and (guidance_mask is not None):
            if len(x.size()) == 2:
                guidance_mask = guidance_mask[:, None]
                guidance_mask = guidance_mask.repeat(1, x.size(1))
            elif len(x.size()) == 4:
                guidance_mask = guidance_mask[:, None, None, None]
                guidance_mask = guidance_mask.repeat(1, x.size(1), x.size(2), x.size(3))
            else:
                raise NotImplementedError()
            guidance_mask = (-1 * (1 - guidance_mask))
            proto = proto * guidance_mask

        if self.conditioning == "stack" and (proto is not None):
            #x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            proto = proto.view(-1, self.latent_size)
            x = torch.cat((proto, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        if self.ae_enc is not None:
            x = self.ae_enc.get_img_from_latent(x, mean=0, std=1, scale=False)
        return x

class Unet1D_MLP_light(nn.Module):
    def __init__(
        self,
        dim,
        latent_size=784,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        conditioning=None,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        norm_type='rms'
    ):
        super().__init__()

        # determine dimensions
        self.latent_size = latent_size
        self.channels = channels
        self.conditioning = conditioning
        #input_channels = channels * (2 if self.conditioning == 'stack' else 1)
        input_channels = latent_size * (2 if self.conditioning == 'stack' else 1)
        init_dim = default(init_dim, dim)
        #self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)
        self.init_conv = nn.Linear(input_channels, init_dim)

        dims = [init_dim, *map(lambda m: int(dim // m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock_MLP, groups = resnet_block_groups)

        # time embeddings

        #time_dim = dim * 4
        time_dim = 128

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            #sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            sinu_pos_emb = SinusoidalPosEmb(128, theta=sinusoidal_pos_emb_theta)
            #fourier_dim = dim
            fourier_dim = 128

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                #block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                #Residual(PreNorm_MLP(dim_in, LinearAttention_MLP(dim_in))),
                Residual(PreNorm_MLP(dim_in, MyAttention(dim_in), norm_type=norm_type)),
                DownsampleMLP(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm_MLP(mid_dim, MyAttention(mid_dim, dim_head = attn_dim_head, heads = attn_heads), norm_type=norm_type))
        #self.mid_attn = Residual(
        #    PreNorm_MLP(mid_dim, LinearAttention_MLP(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                #block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm_MLP(dim_out, MyAttention(dim_out), norm_type=norm_type)),
                UpsampleMLP(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in)
            ]))

        default_out_dim = latent_size * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        #self.final_res_block = block_klass(dim, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Linear(dim, self.out_dim)
        a=1

    def forward(self, x, time, proto = None):

        #x = x.view(-1, 1, self.latent_size)
        #x = x.view(-1, self.latent_size)
        if self.conditioning == "stack" and (proto is not None):
            #x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            proto = proto.view(-1, self.latent_size)
            x = torch.cat((proto, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        #for block1, block2, attn, downsample in self.downs:
        for block2, attn, downsample in self.downs:
            #x = block1(x, t)
            #h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        #for block1, block2, attn, upsample in self.ups:
        for block2, attn, upsample in self.ups:
            #x = torch.cat((x, h.pop()), dim = 1)
            #x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        #x = torch.cat((x, r), dim = 1)

        #x = self.final_res_block(x, t)
        x = self.final_conv(x)
        return x

class Unet1D_MLP_very_light(nn.Module):
    def __init__(
        self,
        dim,
        latent_size=784,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        conditioning=None,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        norm_type='rms'
    ):
        super().__init__()

        # determine dimensions
        self.latent_size = latent_size
        self.channels = channels
        self.conditioning = conditioning
        #input_channels = channels * (2 if self.conditioning == 'stack' else 1)
        input_channels = latent_size * (2 if self.conditioning == 'stack' else 1)
        init_dim = default(init_dim, dim)
        #self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)
        self.init_conv = nn.Linear(input_channels, init_dim)

        dims = [init_dim, *map(lambda m: int(dim // m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock_MLP, groups = resnet_block_groups)

        # time embeddings

        #time_dim = dim * 4
        time_dim = 128

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            #sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            sinu_pos_emb = SinusoidalPosEmb(128, theta=sinusoidal_pos_emb_theta)
            #fourier_dim = dim
            fourier_dim = 128

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                #block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_out, time_emb_dim = time_dim),
                #Residual(PreNorm_MLP(dim_in, LinearAttention_MLP(dim_in))),
                Residual(PreNorm_MLP(dim_out, MyAttention(dim_out), norm_type=norm_type)),
                DownsampleMLP(dim_out, dim_out) if not is_last else nn.Linear(dim_out, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm_MLP(mid_dim, MyAttention(mid_dim, dim_head = attn_dim_head, heads = attn_heads), norm_type=norm_type))
        #self.mid_attn = Residual(
        #    PreNorm_MLP(mid_dim, LinearAttention_MLP(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                #block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm_MLP(dim_out, MyAttention(dim_out), norm_type=norm_type)),
                UpsampleMLP(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in)
            ]))

        default_out_dim = latent_size * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        #self.final_res_block = block_klass(dim, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Linear(dim, self.out_dim)
        a=1

    def forward(self, x, time, proto = None):

        #x = x.view(-1, 1, self.latent_size)
        #x = x.view(-1, self.latent_size)
        if self.conditioning == "stack" and (proto is not None):
            #x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            proto = proto.view(-1, self.latent_size)
            x = torch.cat((proto, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        #for block1, block2, attn, downsample in self.downs:
        for block2, attn, downsample in self.downs:
            #x = block1(x, t)
            #h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        #for block1, block2, attn, upsample in self.ups:
        for block2, attn, upsample in self.ups:
            #x = torch.cat((x, h.pop()), dim = 1)
            #x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        #x = torch.cat((x, r), dim = 1)

        #x = self.final_res_block(x, t)
        x = self.final_conv(x)
        return x


class Unet1D_MLP_easy(nn.Module):
    def __init__(
        self,
        dim,
        latent_size=784,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        conditioning=None,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        norm_type='rms'
    ):
        super().__init__()

        # determine dimensions
        self.latent_size = latent_size
        self.channels = channels
        self.conditioning = conditioning
        #input_channels = channels * (2 if self.conditioning == 'stack' else 1)
        input_channels = latent_size * (2 if self.conditioning == 'stack' else 1)
        init_dim = default(init_dim, dim)
        #self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)
        self.init_conv = nn.Linear(input_channels, init_dim)

        dims = [init_dim, *map(lambda m: dim // m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(Simple_MLP, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                block_klass(dim_in, dim_out, time_emb_dim = time_dim)
            )

        mid_dim = dims[-1]
        self.mid_block = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                block_klass(dim_out + dim_out, dim_in, time_emb_dim = time_dim)
            )

        default_out_dim = latent_size * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Linear(dim, self.out_dim)

    def forward(self, x, time, proto = None):

        #x = x.view(-1, 1, self.latent_size)
        #x = x.view(-1, self.latent_size)
        if self.conditioning == "stack" and (proto is not None):
            #x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            proto = proto.view(-1, self.latent_size)
            x = torch.cat((proto, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1 in self.downs:
            x = block1(x, t)
            h.append(x)

        x = self.mid_block(x, t)

        for block1 in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        return x

class Simple_MLP(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.mlp_1 = nn.Linear(dim+dim_out, dim_out)
        self.mlp_2 = nn.Linear(dim_out, dim_out)
        self.act = nn.GELU()

    def forward(self, x, time_emb = None):
        a=1

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            x = torch.cat([x,time_emb], dim=1)
            #time_emb = rearrange(time_emb, 'b c -> b c 1')
            #scale_shift = time_emb.chunk(2, dim = 1)

        h = self.mlp_1(x)
        h = self.act(h)
        h = self.mlp_2(h)
        return h