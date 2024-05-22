import torch
from .block import Encoder, Decoder
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from .util import DiagonalGaussianDistribution
from .loss_KL import LPIPS_loss_KL, LPIPSWithDiscriminator_KL, Contrastive_Loss_KL
from .loss_VQ import VQLPIPSWithDiscriminator_VQ, LPIPS_loss_VQ

from .quantizer import VectorQuantizer2 as VectorQuantizer

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 loss_type="MSE",
                 loss_white_balance=1.,
                 kl_weight=0.5,
                 perceptual_weight=0.5,
                 learning_rate=1e-4,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 output_vector=False,
                 input_dim=(1, 48, 48),
                 loss_name="LPIPS",
                 disc_start=50001,
                 disc_weight=0.5,
                 disc_num_layers=3,
                 feature_LPIPS='imagenet',
                 device='cuda:0',
                 condition_type=None
                 ):
        super().__init__()
        self.gpu_device=device
        self.save_hyperparameters()
        self.image_key = image_key
        self.ddconfig = ddconfig
        self.embed_dim = embed_dim
        self.condition_type = condition_type
        self.encoder = Encoder(**ddconfig, condition_type=condition_type)
        self.decoder = Decoder(**ddconfig, condition_type=condition_type)
        self.white_balance = loss_white_balance
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.loss_name = loss_name

        assert ddconfig["double_z"]
        self.output_vector = output_vector
        if self.output_vector:
            self.spatial_size = 48//(2**(len(ddconfig["ch_mult"])-1))
            self.nb_channel = ddconfig["z_channels"]
            self.quant_conv = torch.nn.Linear(2 * ddconfig["z_channels"] * self.spatial_size**2, 2 * embed_dim * self.spatial_size**2)
            self.post_quant_conv = torch.nn.Linear(embed_dim * self.spatial_size**2, ddconfig["z_channels"] * self.spatial_size**2)
        else:
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.step = 0
        self.test_step = 0
        if self.loss_name == "LPIPS":
            self.loss = LPIPS_loss_KL(kl_weight=kl_weight,
                                      perceptual_weight=perceptual_weight,
                                      feature_LPIPS=feature_LPIPS)
        elif self.loss_name == "LPIPSDisc":
            self.loss = LPIPSWithDiscriminator_KL(disc_start,
                                                  kl_weight=kl_weight,
                                                  perceptual_weight=perceptual_weight,
                                                  disc_weight=disc_weight,
                                                  disc_in_channels=ddconfig['in_channels'],
                                                  disc_num_layers=disc_num_layers,
                                                  feature_LPIPS=feature_LPIPS)
        elif self.loss_name == "ContrastiveDisc":
            print('Loading Constrastive Loss')
        else :
            raise NotImplementedError()
        with torch.no_grad():
            im = torch.randn(1, *input_dim).to(next(self.parameters()).device)
            if condition_type == 'EncDec':
                condition = torch.randn_like(im)
            else:
                condition = None
            latent = self.get_latent(im, condition=condition)
            self.latent_size = list(latent.size())[1:]

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, condition=None):
        h = self.encoder(x, condition=condition)
        if self.output_vector:
            h = torch.flatten(h, start_dim=1)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments, output_vector=self.output_vector)
        return posterior

    def get_latent(self, x, condition=None, deterministic=False):
        h = self.encoder(x, condition=condition)
        if self.output_vector:
            h = torch.flatten(h, start_dim=1)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments, output_vector=self.output_vector, deterministic=deterministic)
        return posterior.sample()

    def get_scale_factor(self, x, condition=None):
        latent_x = self.get_latent(x, condition=condition).detach()
        return 1. / latent_x.flatten().std()

    def decode(self, z, condition=None):
        z = self.post_quant_conv(z)
        if self.output_vector:
            z = z.resize(z.size(0), self.nb_channel, self.spatial_size, self.spatial_size)
        dec = self.decoder(z, condition=condition)
        return dec

    def forward(self, input, sample_posterior=True, condition=None):
        if self.condition_type == "Dec":
            posterior = self.encode(input, condition=None)
        else:
            posterior = self.encode(input, condition=condition)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, condition=condition)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (inputs, exemplar, label) = batch

        if self.condition_type is not None:
            condition = exemplar
        else:
            condition = None
        if self.loss_name == "LPIPS":
            reconstructions, posterior = self(inputs, condition=condition)
            to_return, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="train")
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        elif self.loss_name == "LPIPSDisc":
            reconstructions, posterior = self(inputs, condition=condition)
            if optimizer_idx == 0:
                # train encoder+decoder+logvar
                aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                to_return = aeloss


            if optimizer_idx == 1:
                # train the discriminator
                discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")

                self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                to_return = discloss


        #self.step += 1

        if (self.global_step % 2000 == 0) and (self.logger is not None):
            self.logger.experiment.log({"Train Input": [wandb.Image(inputs.cpu(), caption="Input Image")],
                                        "Train Reconstruction": [
                                            wandb.Image(reconstructions.cpu(), caption="Reconstructed image")]})

        return to_return

    def validation_step(self, batch, batch_idx):
        (inputs, exemplar, label) = batch
        if self.condition_type is not None:
            condition = exemplar
        else:
            condition = None
        reconstructions, posterior = self(inputs, condition=condition)

        if self.loss_name == "LPIPS":
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="test")
            self.log_dict(log_dict_ae)
        elif self.loss_name == "LPIPSDisc":
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)


        self.test_step += 1
        if (self.test_step % 250 == 0):
            self.logger.experiment.log({"Test Input": [wandb.Image(inputs.cpu(), caption="Input image")],
                                        "Test Reconstruction": [wandb.Image(reconstructions.cpu(), caption="Reconstructed image")]})

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        if self.loss_name == "LPIPS":
            return [opt_ae], []
        elif self.loss_name == "LPIPSDisc":
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], []



    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, xrec, x):
        log = dict()
        log["reconstructions"] = xrec
        log["inputs"] = x
        return log


class AutoencoderKL_Contrastive(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.loss_name == 'ContrastiveDisc':
            self.loss = Contrastive_Loss_KL(kl_weight=self.kl_weight, device=self.gpu_device)
            self.automatic_optimization = False
        else:
            raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        #loss, log_disc = self.loss(inputs, reconstructions, posterior, loss_type='discriminator', split='val')
        #loss, log_gen = self.loss(inputs, reconstructions, posterior, loss_type='generator', split='val')
        pass

    def training_step(self, batch, batch_idx):
        (inputs, exemplar, label) = batch
        opt_encoder, opt_decoder, opt_shared, opt_disc_head, opt_cont_head = self.optimizers()

        ## train discriminator
        opt_shared.zero_grad()
        opt_disc_head.zero_grad()

        z = torch.randn(inputs.size(0), 144).to(inputs.device)
        gen_imgs = self.decode(z)

        reconstructions, posterior = self(inputs)
        loss, log_disc = self.loss(inputs, reconstructions, posterior, loss_type='discriminator', split='train', gen_imgs=gen_imgs)
        self.manual_backward(loss)
        opt_shared.step()
        opt_disc_head.step()
        self.log_dict(log_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        ## train generator
        if self.global_step % 5 == 0:
            ## cal loss
            opt_decoder.zero_grad()
            opt_encoder.zero_grad()
            z = torch.randn(128, 144).to(inputs.device)
            gen_imgs = self.decode(z)
            reconstructions, posterior = self(inputs)
            loss, log_gen = self.loss(inputs, reconstructions, posterior, loss_type='generator', split='train', gen_imgs=gen_imgs)
            self.manual_backward(loss)
            opt_decoder.step()
            opt_encoder.step()
            self.log_dict(log_gen, prog_bar=False, logger=True, on_step=True, on_epoch=False)

            ## contrastive loss
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            opt_shared.zero_grad()
            opt_cont_head.zero_grad()

            reconstructions, posterior = self(inputs)
            loss, log_contras, d_k = self.loss(inputs, reconstructions, posterior, loss_type='contrastive', split='train')
            self.manual_backward(loss)
            opt_encoder.step()
            opt_decoder.step()
            opt_shared.step()
            opt_cont_head.step()
            self.loss.move_pointer(inputs, d_k)
            self.log_dict(log_contras, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        if self.global_step % 100 == 0:
            with torch.no_grad():
                rec_pix = torch.nn.MSELoss()(inputs, reconstructions ).mean()
                to_log = {"train/mse_rec": rec_pix.detach().mean()}

            self.log_dict(to_log, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        if (self.global_step % 2000 == 0) and (self.logger is not None):
            self.logger.experiment.log({"Train Input": [wandb.Image(inputs.cpu(), caption="Input Image")],
                                        "Train Reconstruction": [
                                            wandb.Image(reconstructions.cpu(), caption="Reconstructed image")]})

    def configure_optimizers(self):
        lr = self.learning_rate
        encoder_param = list(self.encoder.parameters()) + \
                        list(self.quant_conv.parameters())
        opt_encoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder_param),
            lr, (0, 0.9))

        decoder_param = list(self.decoder.parameters()) + list(self.post_quant_conv.parameters())
        opt_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, decoder_param),
            lr, (0, 0.9))

        shared_params = list(self.loss.dual_encoder.block1.parameters()) + \
                        list(self.loss.dual_encoder.block2.parameters()) + \
                        list(self.loss.dual_encoder.block3.parameters()) + \
                        list(self.loss.dual_encoder.block4.parameters()) + \
                        list(self.loss.dual_encoder.l5.parameters())
        opt_shared = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                             shared_params),
                                      lr, (0, 0.9))

        opt_disc_head = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                self.loss.dual_encoder.head_disc.parameters()),
                                         lr, (0, 0.9))

        cont_params = list(self.loss.dual_encoder.head_b1.parameters()) + \
                      list(self.loss.dual_encoder.head_b2.parameters()) + \
                      list(self.loss.dual_encoder.head_b3.parameters()) + \
                      list(self.loss.dual_encoder.head_b4.parameters())
        opt_cont_head = torch.optim.Adam(filter(lambda p: p.requires_grad, cont_params),
                                         lr, (0, 0.9))

        return opt_encoder, opt_decoder, opt_shared, opt_disc_head, opt_cont_head

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 learning_rate=1e-4,
                 loss_type="MSE",
                 white_balance=1.0,
                 perceptual_weight=1.0,
                 l1_sparsity=0.0,
                 vq_weight=0.1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 input_dim=(1, 48, 48),
                 lr_g_factor=1.0,
                 loss_name="LPIPS",
                 disc_start=50001,
                 disc_weight=0.5,
                 disc_num_layers=2,
                 beta_vq=0.25,
                 feature_LPIPS='imagenet',
                 condition_type=None
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.ddconfig = ddconfig
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.lr_g_factor = lr_g_factor
        self.learning_rate = learning_rate
        self.condition_type = condition_type
        self.encoder = Encoder(**ddconfig, condition_type=condition_type)
        self.decoder = Decoder(**ddconfig, condition_type=condition_type)
        self.loss_name = loss_name
        self.beta_vq = beta_vq
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=self.beta_vq,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.loss_type = loss_type
        self.vq_weight = vq_weight
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.white_balance = white_balance

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config

        self.step = 0
        self.test_step = 0
        if self.loss_name == "LPIPS":
            self.loss = LPIPS_loss_VQ(codebook_weight=vq_weight,
                                      perceptual_weight=perceptual_weight,
                                      n_classes=n_embed,
                                      feature_LPIPS=feature_LPIPS,
                                      l1_sparsity=l1_sparsity)
        elif self.loss_name == "LPIPSDisc":
            self.loss = VQLPIPSWithDiscriminator_VQ(disc_start,
                                                    codebook_weight=vq_weight,
                                                    perceptual_weight=perceptual_weight,
                                                    disc_weight=disc_weight,
                                                    disc_in_channels=ddconfig['in_channels'],
                                                    n_classes=n_embed,
                                                    disc_num_layers=disc_num_layers,
                                                    feature_LPIPS=feature_LPIPS,
                                                    l1_sparsity=l1_sparsity)
        else:
            raise NotImplementedError()
        with torch.no_grad():
            im = torch.randn(1, *input_dim).to(next(self.parameters()).device)
            if condition_type == 'EncDec':
                condition = torch.randn_like(im)
            else:
                condition = None
            latent = self.get_latent(im, condition=condition)
            self.latent_size = list(latent.size())[1:]

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x, condition=None):
        h = self.encoder(x, condition=condition)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h

    def get_latent(self, x, condition=None, deterministic=False):
        h = self.encoder(x, condition=condition)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        return quant

    def get_scale_factor(self, x, condition=None):
        latent_x = self.get_latent(x, condition=condition).detach()
        return 1. / latent_x.flatten().std()

    def encode_to_prequant(self, x, condition=None):
        h = self.encoder(x, condition=condition)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, condition=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, condition=condition)
        return dec

    def decode_code(self, code_b, condition=None):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, condition=condition)
        return dec

    def forward(self, input, return_pred_indices=False, condition=None):
        if self.condition_type == "Dec":
            quant, diff, (_, _, ind), h = self.encode(input, condition=None)
        else:
            quant, diff, (_, _, ind), h = self.encode(input, condition=condition)
        dec = self.decode(quant, condition=condition)
        if return_pred_indices:
            return dec, diff, ind, h
        return dec, diff, h

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    #def rec_loss(self, inputs, reconstructions, white_balance=1.0):
    #    if self.loss_type == 'MSE':
    #        rec_loss_in1 = (inputs.contiguous() - reconstructions.contiguous()) + torch.abs(
    #            inputs.contiguous() - reconstructions.contiguous())
    #        rec_loss_in0 = (inputs.contiguous() - reconstructions.contiguous()) - torch.abs(
    #            inputs.contiguous() - reconstructions.contiguous())

    #        rec_loss_in1 = rec_loss_in1 / 2
    #        rec_loss_in0 = -1 * rec_loss_in0 / 2

    #        rec_loss_in1 = rec_loss_in1 ** 2
    #        rec_loss_in0 = rec_loss_in0 ** 2

    #        rec_loss = rec_loss_in0 + white_balance * rec_loss_in1
    #        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

    #    elif self.loss_type == 'BCE':
    #        rec_loss = F.binary_cross_entropy(reconstructions, inputs, reduction='none')
    #        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
    #    return rec_loss

    #def vq_loss(self, inputs, reconstruction, qloss, white_balance=1.0, split="train"):
    #    rec_loss = self.rec_loss(inputs, reconstruction, white_balance)
    #    loss = rec_loss + self.vq_weight * qloss
    #    log_dict = {"{}/total_loss".format(split): loss.clone().detach(),
    #                "{}/rec_loss".format(split): rec_loss.detach(),
    #                "{}/qloss".format(split): qloss.detach()}
    #    return loss, log_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        (inputs, exemplar, label) = batch

        if self.condition_type is not None:
            condition = exemplar
        else:
            condition = None

        xrec, qloss, ind, h = self(inputs, return_pred_indices=True, condition=condition)
        if self.loss_name == "LPIPS":
            to_return, log_dict_ae = self.loss(qloss, inputs, xrec, split="train", predicted_indices=ind, sparse_repr=h)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        elif self.loss_name == "LPIPSDisc":
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(qloss, inputs, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",
                                                predicted_indices=ind,
                                                sparse_repr=h)

                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                to_return = aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(qloss, inputs, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",
                                                    sparse_repr=h)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                to_return = discloss

        #self.step += 1
        if (self.global_step % 2000 == 0) and (self.logger is not None):
            self.logger.experiment.log({"Train Input": [wandb.Image(inputs.cpu(), caption="Input Image")],
                                        "Train Reconstruction": [
                                            wandb.Image(xrec.cpu(), caption="Reconstructed image")]
                                        })

        #self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return to_return

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        (inputs, exemplar, label) = batch
        if self.condition_type is not None:
            condition = exemplar
        else:
            condition = None

        xrec, qloss, ind, h = self(inputs, return_pred_indices=True, condition=condition)
        if self.loss_name == "LPIPS":
            aeloss, log_dict_ae = self.loss(qloss, inputs, xrec, split="val" + suffix, predicted_indices=ind, sparse_repr=h)
            self.log_dict(log_dict_ae)

        elif self.loss_name == "LPIPSDisc":
            aeloss, log_dict_ae = self.loss(qloss, inputs, xrec, 0,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=ind
                                            )

            discloss, log_dict_disc = self.loss(qloss, inputs, xrec, 1,
                                                self.global_step,
                                                last_layer=self.get_last_layer(),
                                                split="val" + suffix,
                                                predicted_indices=ind,
                                                sparse_repr=h
                                                )
            rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
            self.log(f"val{suffix}/rec_loss", rec_loss,
                     prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val{suffix}/aeloss", aeloss,
                     prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            #if version.parse(pl.__version__) >= version.parse('1.4.0'):
            if int(pl.__version__.replace(".", "")) > int("1.4.0".replace(".", "")):
                del log_dict_ae[f"val{suffix}/rec_loss"]
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)

        self.test_step += 1
        if (self.test_step % 250 == 0):
            self.logger.experiment.log({"Test Input": [wandb.Image(inputs.cpu(), caption="Input image")],
                                        "Test Reconstruction": [wandb.Image(xrec.cpu(), caption="Reconstructed image")]})


        return self.log_dict



    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate

        print("lr_g", lr_g)

        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        if self.loss_name == "LPIPS":
            return [opt_ae], []

        elif self.loss_name == "LPIPSDisc":
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x