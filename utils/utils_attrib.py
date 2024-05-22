import torch.nn as nn
import torch
import numpy as np
from models.AutoEncoder.constrative_tools import pairwise_distances

class OneShotLinearProb(nn.Module):
    def __init__(self, ae, args_ae):
        super().__init__()
        self.args_ae = args_ae
        self.encoder = ae.encoder
        self.probe = nn.Sequential(nn.Flatten(start_dim=1),
                      nn.Linear(np.prod(args_ae.latent_size), 128))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, image, proto):

        if self.args_ae.w_kl != 0:
            q_mean_im, q_logvar_im = self.encoder(image)
            q_mean_proto, q_logvar_proto= self.encoder(proto)
            feat_image = self.reparametrize(q_mean_im, q_logvar_im)
            feat_proto = self.reparametrize(q_mean_proto, q_logvar_proto)
        else:
            feat_image = self.encoder(image)
            feat_proto = self.encoder(proto)
        z_image = self.probe(feat_image)
        z_proto = self.probe(feat_proto)
        all_pairs_distance = pairwise_distances(z_image, z_proto)
        return all_pairs_distance

class probe_AE(nn.Module):
    def __init__(self, ae, args_ae):
        super().__init__()
        self.args_ae = args_ae
        self.encoder = ae.encoder
        self.probe = nn.Sequential(nn.Flatten(start_dim=1),
                      nn.Linear(np.prod(args_ae.latent_size), 115))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, image):

        if self.args_ae.w_kl != 0:
            q_mean, q_logvar = self.encoder(image)
            feat = self.reparametrize(q_mean, q_logvar)
        else:
            feat = self.encoder(image)
        logits = self.probe(feat)
        return logits