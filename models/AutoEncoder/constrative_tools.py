import torch.nn as nn
import torch
import torch.nn.functional as F
gf_dim = 256
df_dim = 128
g_spectral_norm = False
d_spectral_norm = True
bottom_width = 4



def weights_init(m):
    init_type="xavier_uniform"
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class DisBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedDisBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class DualEncoder(nn.Module):
    def __init__(self, cont_dim=16, activation=nn.ReLU()):
        super(DualEncoder, self).__init__()
        self.ch = df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(1, self.ch)
        self.block2 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.head_disc = nn.utils.spectral_norm(nn.Linear(cont_dim, 1))
        self.l5 = nn.Linear(self.ch, cont_dim, bias=False)
        self.l5 = nn.utils.spectral_norm(self.l5)
        self.head_b1 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(576, cont_dim, bias=False)
        )
        self.head_b2 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(144, cont_dim, bias=False)
        )
        self.head_b3 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(144, cont_dim, bias=False)
        )
        self.head_b4 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(144, cont_dim, bias=False)
        )

    def forward(self, x, mode="dual"):
        h = x
        h1 = self.block1(h)  # 128x24x24
        h2 = self.block2(h1)  # 128x12x12
        h3 = self.block3(h2)  # 128x12x12
        h4 = self.block4(h3)  # 128x12x12
        h = self.activation(h4)
        h = h.sum(2).sum(2)
        h = self.l5(h)
        disc_out = self.head_disc(h)
        if mode == "dis":
            return disc_out
        elif mode == "cont":
            cont_out = {
                "b1-raw": h1,
                "b2-raw": h2,
                "b3-raw": h3,
                "b4-raw": h4,
                "b1": self.head_b1(h1),
                "b2": self.head_b2(h2),
                "b3": self.head_b3(h3),
                "b4": self.head_b4(h4),
                "final": h
            }
            return cont_out
        elif mode == "cont_local":
            cont_out = {
                "local_h1": h1,  # 128x16x16
                "local_h2": h2,  # 128x8x8
                "local_h3": h3,  # 128x8x8
                "local_h4": h4,  # 128x8x8
                "b1": self.head_b1(h1),
                "final": h
            }
            return cont_out

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].ravel().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def pairwise_distances(x, y):
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    distances = (
            x.unsqueeze(1).expand(n_x, n_y, -1) -
            y.unsqueeze(0).expand(n_x, n_y, -1)
    ).pow(2).sum(dim=2)
    return distances

def create_fake_label(label):
    new_label = torch.zeros_like(label)
    unique_elem = torch.unique(label)
    count = 0
    for each_elem in unique_elem:
        filter = label == each_elem
        new_label[filter] = count
        count += 1
    return new_label

class proto_net_loss(nn.Module):
    def __init__(self, latent_size, metric_layer_size=128):
        super().__init__()
        self.latent_size = latent_size
        self.projector = nn.Sequential(
            nn.Linear(latent_size, metric_layer_size, bias=False),
            nn.BatchNorm1d(metric_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(metric_layer_size, metric_layer_size, bias=False)
        )

    def forward(self, variations, exemplar):

        #batch_size = latent1.size(0)
        z1 = self.projector(variations)
        z2 = self.projector(exemplar)
        all_pairs_distance = pairwise_distances(z1, z2)
        #diag_dist = torch.diagonal(all_pairs_distance).mean()
        #off_diag_dist = off_diagonal(all_pairs_distance).mean()
        #return diag_dist - off_diag_dist/2
        return all_pairs_distance

class barlow_loss(nn.Module):
    def __init__(self, latent_size, lambda_barlow=0.005):
        super().__init__()
        self.latent_size = latent_size
        self.lambda_barlow = lambda_barlow
        self.projector = nn.Sequential(
            nn.Linear(latent_size, latent_size, bias=False),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False)
        )
        self.bn = nn.BatchNorm1d(latent_size, affine=False)

    def forward(self, latent1, latent2):
        batch_size = latent1.size(0)
        z1 = self.projector(latent1)
        z2 = self.projector(latent2)
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_barlow * off_diag
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss