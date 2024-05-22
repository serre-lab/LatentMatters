import torch
import torch.nn as nn
import torch.nn.functional as F
from .constrative_tools import weights_init, DualEncoder, accuracy

from taming.modules.losses.vqperceptual import *
from .custom_LPIPS import LPIPS_QD

class LPIPS_loss_KL(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, perceptual_weight=1.0,
                 feature_LPIPS='imagenet'):
        super().__init__()
        self.kl_weight = kl_weight
        if feature_LPIPS == 'imagenet':
            self.perceptual_loss = LPIPS().eval()
        elif feature_LPIPS == 'quickdraw':
            print(f"running with LPIPS on quickdraw")
            self.perceptual_loss = LPIPS_QD().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, split="train", weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log

class Contrastive_Loss_KL(nn.Module):
    def __init__(self,
                 kl_weight=1.0,
                 cont_weight=1.0,
                 cont_dim=16,
                 layers=("b1", "final"),
                 cont_k=8192,
                 cont_temp=0.07,
                 device='cuda:0'):

        super().__init__()
        self.kl_weight = kl_weight
        self.dual_encoder = DualEncoder(cont_dim)
        self.dual_encoder.apply(weights_init)
        self.dual_encoder_M = DualEncoder(cont_dim)
        for p, p_momentum in zip(self.dual_encoder.parameters(), self.dual_encoder_M.parameters()):
            p_momentum.data.copy_(p.data)
            p_momentum.requires_grad = False
        self.d_queue, self.d_queue_ptr = {}, {}
        self.layers = layers
        self.cont_temp = cont_temp
        self.cont_k = cont_k
        self.lambda_cont = cont_weight / len(layers)
        for layer in self.layers:
            self.d_queue[layer] = torch.randn(cont_dim, cont_k).to(device)
            self.d_queue[layer] = F.normalize(self.d_queue[layer], dim=0)
            self.d_queue_ptr[layer] = torch.zeros(1, dtype=torch.long)

    #def forward(self, inputs, reconstructions, posteriors, optimizer_idx):
    def forward(self, inputs, reconstructions, posterior, loss_type, split="train", gen_imgs=None):
        if loss_type == 'discriminator':
            if gen_imgs is not None :
                gen_imgs = gen_imgs.detach()
                fake_validity = self.dual_encoder(gen_imgs, mode="dis")
            real_validity = self.dual_encoder(inputs, mode="dis")
            rec_validity = self.dual_encoder(reconstructions, mode="dis")
            if gen_imgs is not None :
                d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                         torch.mean(nn.ReLU(inplace=True)(1.0 + fake_validity)) * 0.5 + \
                         torch.mean(nn.ReLU(inplace=True)(1.0 + rec_validity)) * 0.5
            else:
                d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1.0 + rec_validity))*0.5
            log = {"{}/d_loss".format(split): d_loss.detach().clone().detach().mean(),
                   }
            return d_loss, log

        elif loss_type == 'generator':
            if gen_imgs is not None:
                fake_validity = self.dual_encoder(gen_imgs, mode="dis")
            rec_validity = self.dual_encoder(reconstructions, mode="dis")
            if gen_imgs is not None:
                g_loss = -(torch.mean(fake_validity) * 0.5 + torch.mean(rec_validity) * 0.5)
            else:
                g_loss = -(torch.mean(rec_validity) * 0.5)
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            cal_loss = g_loss + self.kl_weight*kl_loss
            log = {"{}/cal_loss".format(split): cal_loss.clone().detach().mean(),
                   "{}/kl_loss".format(split): kl_loss.clone().detach().mean(),
                   "{}/g_loss".format(split): g_loss.clone().detach().mean()
                   }
            return cal_loss, log

        elif loss_type == 'contrastive':
            im_k = inputs
            im_q = reconstructions
            with torch.no_grad():
                # update momentum encoder
                for p, p_mom in zip(self.dual_encoder.parameters(), self.dual_encoder_M.parameters()):
                    p_mom.data = (p_mom.data * 0.999) + (p.data * (1.0 - 0.999))
                d_k = self.dual_encoder_M(im_k, mode="cont")
                for l in self.layers:
                    d_k[l] = F.normalize(d_k[l], dim=1)
            total_cont = torch.tensor(0.0).to(inputs.device)
            d_q = self.dual_encoder(im_q, mode="cont")
            for l in self.layers:
                q = F.normalize(d_q[l], dim=1)
                k = d_k[l]
                queue = self.d_queue[l]
                l_pos = torch.einsum("nc,nc->n", [k, q]).unsqueeze(-1)
                l_neg = torch.einsum('nc,ck->nk', [q, queue.detach()])
                logits = torch.cat([l_pos, l_neg], dim=1) / self.cont_temp
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(inputs.device)
                cont_loss = nn.CrossEntropyLoss()(logits, labels) * self.lambda_cont
                total_cont += cont_loss
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = kl_loss+total_cont
            log = {"{}/cont_loss".format(split): loss.clone().detach().mean(),
                   "{}/acc1".format(split): acc1.item(),
                   "{}/acc5".format(split): acc5.item(),
                   }

            return loss, log, d_k
        else:
            NotImplementedError()

    def move_pointer(self, inputs, d_k):
        with torch.no_grad():
            for l in self.layers:
                ptr = int(self.d_queue_ptr[l])
                self.d_queue[l][:, ptr:(ptr + inputs.shape[0])] = d_k[l].transpose(0, 1)
                ptr = (ptr + inputs.shape[0]) % self.cont_k  # move the pointer ahead
                self.d_queue_ptr[l][0] = ptr


class LPIPSWithDiscriminator_KL(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", feature_LPIPS='imagenet'):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        if feature_LPIPS == 'imagenet':
            self.perceptual_loss = LPIPS().eval()
        elif feature_LPIPS == 'quickdraw':
            print(f"running with LPIPS on quickdraw")
            self.perceptual_loss = LPIPS_QD().eval()

        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log