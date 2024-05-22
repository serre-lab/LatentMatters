import torchvision
from models.AutoEncoder.one_d_AE import One_d_AE, RAE_48x48
from models.AutoEncoder.one_d_AE import AE as identity_ae
import torch
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torchvision.transforms import transforms
import torch.nn as nn
from utils.monitoring import plot_img, make_directories, make_grid
#from utils.monitoring import make_directories, make_grid
import os
from models.parser import dataset_args, training_args, diffusion_args, exp_args
from models.Diffusion.one_d_diffusion import One_d_UNet, Unet1D, Unet1D_MLP, Unet1D_MLP_easy, Unet1D_MLP_light, Unet1D_MLP_very_light
#from models.Diffusion.sampler import DDPM, DDPM_ELBO
from models.Diffusion.one_d_diffuser import DDPM_ELBO
from utils.dataloader import load_dataset_exemplar
import matplotlib.pyplot as plt
import argparse
import wandb
import numpy as np
import random
from RecOri.utils import generate_samples_1d
from argparse import Namespace

parser = argparse.ArgumentParser()
parser = dataset_args(parser)
parser = training_args(parser)
parser = diffusion_args(parser)
parser = exp_args(parser)
args = parser.parse_args()

if args.model_name == 'ldm_ddpm1d_stack':
    wb_name = "LDM_1D_Diffusion"
elif args.model_name == 'ldm_cfgdm1d_stack':
    wb_name = "G_LDM_1D_Diffusion"
if args.dataset == 'quickdraw_clust':
    wb_name += '_qd'
if args.device == 'meso':
    wb_name += '_osc'

if args.device == 'meso':
    args.device = 'cuda:'+str(torch.cuda.current_device())

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

kwargs = {'preload': args.preload,
          'drop_last': True}
train_loader, test_loader, args = load_dataset_exemplar(args, shape=args.input_shape, **kwargs)

one_test_batch = next(iter(test_loader))
test_image, test_exemplar = one_test_batch[0].to(args.device), one_test_batch[1].to(args.device)
one_train_batch = next(iter(train_loader))
train_image, train_exemplar = one_train_batch[0].to(args.device), one_train_batch[1].to(args.device)

#### Load the AE
if args.ae_name != '':
    path_to_autoenc = os.path.join(args.ae_path, args.ae_name)
    args_ae = torch.load(path_to_autoenc + '/param.config')
    weight_ae = torch.load(path_to_autoenc + '/_last.model', map_location=torch.device('cpu'))
    #weight_ae = torch.load(path_to_autoenc + '/ep_40.model', map_location=torch.device('cpu'))
    AE = RAE_48x48(**vars(args_ae))
    AE.load_state_dict(weight_ae, strict=False)
    resnet_block_groups = 8
    ae_diff = None
    #if AE.vq_reg is not None:
    #    args_ae.latent_size = args_ae.latent_size*4
else:
    #args_ae = {"latent_size": 256}
    #args_ae = Namespace(**args_ae)
    ae_diff = RAE_48x48(latent_size=256,
                   ae_dim=128,
                   tanh=False,
                   k=0,
                   w_kl=0,
                   w_cons=0,
                   vq_reg=None,
                   commit_vq=0,
                   m=0.99).to(args.device)
    resnet_block_groups = 8
    args_ae = {"latent_size": 256}
    args_ae = Namespace(**args_ae)
    #resnet_block_groups = min(args.unet_dim, 8)
    AE = identity_ae()

AE = AE.to(args.device)
AE.eval()
######## Estimate std and mean of the AE latent (on one batch)
#one_test_batch = next(iter(train_loader))
#train_image_stats = one_test_batch[0].to(args.device)
if args.ae_name == '':
    mean, std = 0, 1
else:
    mean, std = AE.estimate_stats(train_image)
#### Load the diffusion model
if args.model_name == 'ldm_ddpm1d_stack':
    conditioning = "stack"
    drop_out = 0
elif args.model_name == 'ldm_cfgdm1d_stack':
    drop_out = 0.1
    conditioning = "stack"
else:
    conditioning = None
    drop_out=0

if args.diffuser == 'conv1d_unet':
    diffuser = Unet1D
elif args.diffuser == 'att_mlp_1d':
    diffuser = Unet1D_MLP
elif args.diffuser == 'att_mlp_1d_light':
    diffuser = Unet1D_MLP_light
elif args.diffuser == 'att_mlp_1d_very_light':
    diffuser = Unet1D_MLP_very_light
elif args.diffuser == 'mlp_1d':
    diffuser = Unet1D_MLP_easy
else:
    raise NotImplementedError

denoiser_model = diffuser(dim=args.unet_dim,
                                latent_size=args_ae.latent_size,
                                dim_mults=args.unet_mult,
                                channels=args.input_shape[0],
                                attn_dim_head=args.attn_dim_head,
                                attn_heads=args.attn_heads,
                                conditioning=conditioning,
                                resnet_block_groups=resnet_block_groups,
                                norm_type=args.norm_type,
                                ae_enc=ae_diff
                                )

#t = torch.linspace(0, 1000, 128).long()
#img = torch.randn(128, 128)

#out = denoiser_model(img, t, img)

#stop
diffusion_model = DDPM_ELBO(nn_model=denoiser_model,
                       betas=(args.betas[0], args.betas[1]),
                       n_T=args.timestep,
                       device=args.device,
                       drop_prob=drop_out).to(args.device)

args = make_directories(args, ae_args=args_ae)
if not args.debug:
    wandb.init(project=wb_name, config=vars(args), entity='victorboutin')
    wandb.run.name = args.model_signature
    wandb.run.save()
    torch.save(args, args.snap_dir + 'param.config')

#if args.ae_name != '':
print('# param : {0:,}'.format(sum(p.numel() for p in diffusion_model.parameters())))
optimizer = AdamW(diffusion_model.parameters(), lr=args.learning_rate, eps=1e-5)
#else:
#    all_model_param = list(diffusion_model.parameters()) + list(AE.parameters())
#    print('# param : {0:,}'.format(sum(p.numel() for p in all_model_param)))
#    optimizer = AdamW(all_model_param, lr=args.learning_rate, eps=1e-5)

if args.scheduler == "one_cycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                 steps_per_epoch=len(train_loader), epochs=args.epoch, pct_start=0.1)
elif args.scheduler == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

if False:
    one_image = test_image[0:1].to(args.device)
    latent = AE.get_latent_from_img(one_image, mean=mean, std=std)
    latent_noisy = diffusion_model.test_noise(latent, 25)
    decoded = AE.get_img_from_latent(latent_noisy, mean=mean, std=std)
    plot_img(decoded)
    #fig = plt.figure(figsize=(50, 50))
    #for i in range(25):
    #    ax = fig.add_subplot(5, 5, i + 1)
    #    ax.hist(latent_noisy[i].cpu().detach(), bins=100)
    #plt.show()
#x_noisy = diffusion_model.test_noise(one_image, 100)
#plot_img(x_noisy.view(-1, 1, 28, 28), nrow=10, ncol=10, scale_each=True)


for ep in range(args.epoch):
    denoiser_model.train()
    pbar = tqdm(train_loader)
    for idx_batch, (image, prototype, label) in enumerate(pbar):
        image, prototype = image.to(args.device), prototype.to(args.device)
        s_latent = AE.get_latent_from_img(image, mean=mean, std=std)
        if conditioning == "stack":
            s_latent_proto = AE.get_latent_from_img(prototype, mean=mean, std=std)
        else:
            s_latent_proto = None
        loss = diffusion_model(s_latent, s_latent_proto)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scheduler == "one_cycle":
            scheduler.step()
        pbar.set_description(f"{ep+1}/{args.epoch} loss_tr: {loss:.4f}")
    if args.scheduler == "step":
        scheduler.step()
    if not args.debug:
        wandb.log({"train/loss": loss,
                   "epoch": ep,
                   "lr": optimizer.param_groups[0]['lr']})
    denoiser_model.eval()
    if ep % 5 == 0:
        if conditioning == 'stack':
            ## get latent for proto
            lt_proto_test = AE.get_latent_from_img(test_exemplar, mean=mean, std=std)
            lt_proto_train = AE.get_latent_from_img(train_exemplar, mean=mean, std=std)

            ## get_latent for image:
            lt_img_test = AE.get_latent_from_img(test_image, mean=mean, std=std)
            lt_img_train = AE.get_latent_from_img(train_image, mean=mean, std=std)

            reco_tr_lt, noisy_latent_tr = diffusion_model.partial_sample_c(lt_img_train, lt_proto_train, nb_it=225)
            reco_te_lt, noisy_latent_te = diffusion_model.partial_sample_c(lt_img_test, lt_proto_test, nb_it=225)

            l2_loss_tr = ((lt_img_train - reco_tr_lt) ** 2).sum(dim=1).mean()
            l2_loss_te = ((lt_img_test - reco_te_lt) ** 2).sum(dim=1).mean()

            init_tr = AE.get_img_from_latent(noisy_latent_tr, mean=mean, std=std)
            init_te = AE.get_img_from_latent(noisy_latent_te, mean=mean, std=std)
            reco_tr = AE.get_img_from_latent(reco_tr_lt, mean=mean, std=std)
            reco_te = AE.get_img_from_latent(reco_te_lt, mean=mean, std=std)

            to_show_tr = torch.cat([train_image[0:10], init_tr[0:10], reco_tr[0:10],
                                    train_image[10:20], init_tr[10:20], reco_tr[10:20]])
            to_show_te = torch.cat([test_image[0:10], init_te[0:10], reco_te[0:10],
                                    test_image[10:20], init_te[10:20], reco_te[10:20]])

            if not args.debug:
                #img_te = make_grid(sample_te, ncol=10, nrow=10, normalize=True, scale_each=True)
                img_te = make_grid(to_show_te, ncol=10, nrow=10, normalize=True, scale_each=True)
                #img_tr = make_grid(sample_tr, ncol=10, nrow=10, normalize=True, scale_each=True)
                img_tr = make_grid(to_show_tr, ncol=10, nrow=10, normalize=True, scale_each=True)
                reco_te = wandb.Image(img_te, caption='reco at ep:{}'.format(ep))
                reco_tr = wandb.Image(img_tr, caption='reco at ep:{}'.format(ep))

                fig, ax = plt.subplots()
                ax.hist(torch.ravel(lt_img_train).cpu().detach().numpy(), bins=100, density=True, color='red')
                ax.hist(torch.ravel(reco_tr_lt).cpu().detach().numpy(), bins=100, density=True, color='blue', alpha=0.5)
                ax.set_xlim([-5, 5])
                ax.set_ylim([0, 1])

                fig1, ax1 = plt.subplots()
                ax1.hist(torch.ravel(lt_img_test).cpu().detach().numpy(), bins=100, density=True, color='red')
                ax1.hist(torch.ravel(reco_te_lt).cpu().detach().numpy(), bins=100, density=True, color='blue', alpha=0.5)
                ax1.set_xlim([-5, 5])
                ax1.set_ylim([0, 1])

                wandb.log({"train/reco": reco_tr,
                           "test/reco": reco_te,
                           "train/loss_reco": l2_loss_tr,
                           "test/loss_reco": l2_loss_te,
                           "train/distribution": wandb.Image(fig),
                           "test/distribution": wandb.Image(fig1)})
                plt.close()

    if ep % 5 == 0:
        if conditioning is not None:
            lt_proto_test = AE.get_latent_from_img(test_exemplar[0:10], mean=mean, std=std)
            lt_proto_test = lt_proto_test.repeat(10, 1)
            #lt_proto_test = lt_proto_test.repeat(10, 1, 1, 1)

            lt_proto_train = AE.get_latent_from_img(train_exemplar[0:10], mean=mean, std=std)
            lt_proto_train = lt_proto_train.repeat(10, 1)
            #lt_proto_train = lt_proto_train.repeat(10, 1, 1, 1)

            if args.ae_name == '':
                size_sample = args.input_shape
            else:
                size_sample = (args_ae.latent_size,)
            sample_te = diffusion_model.sample_c(image_size=size_sample,
                                                 batch_size=100,
                                                 cond=lt_proto_test)
            sample_te = AE.get_img_from_latent(sample_te, mean=mean, std=std)
            sample_tr = diffusion_model.sample_c(image_size=size_sample,
                                                 batch_size=100,
                                                 cond=lt_proto_train)
            sample_tr = AE.get_img_from_latent(sample_tr, mean=mean, std=std)

        else:
            sample = diffusion_model.sample_c(image_size=(args_ae.latent_size,),
                                          batch_size=100,
                                          cond=None)
            sample = AE.get_img_from_latent(sample, mean=mean, std=std)
        if not args.debug:
            if conditioning is not None:
                sample_te = sample_te.view(100, *args.input_shape).cpu()
                sample_tr = sample_tr.view(100, *args.input_shape).cpu()
                to_show_te = torch.cat([test_exemplar[0:10].cpu(), sample_te.cpu()], dim=0)
                to_show_tr = torch.cat([train_exemplar[0:10].cpu(), sample_tr.cpu()], dim=0)
                img_te = make_grid(to_show_te, ncol=10, nrow=11, normalize=True, scale_each=True)
                img_tr = make_grid(to_show_tr, ncol=10, nrow=11, normalize=True, scale_each=True)
                gene_te = wandb.Image(img_te, caption='generation at ep:{}'.format(ep))
                gene_tr = wandb.Image(img_tr, caption='generation at ep:{}'.format(ep))
                wandb.log({"train/gene": gene_tr,
                           "test/gene": gene_te})

            else:
                sample = sample.view(100, *args.input_shape).cpu()
                img = make_grid(sample, ncol=10, nrow=100, normalize=True, scale_each=True)
                gene_tr = wandb.Image(img, caption='generation at ep:{}'.format(ep))
                wandb.log({"train/gene": gene_tr})
    if ep % (int(args.epoch / 2)) == 0:
        if not args.debug:
            torch.save(diffusion_model.state_dict(), args.snap_dir + f'ep_{ep}.model')

if not args.debug:
    torch.save(diffusion_model.state_dict(), args.snap_dir + '_last.model')
    if args.ae_name == '':
        torch.save(AE.state_dict(), args.snap_dir + f'_last_ae.model')

if args.generate_img:
    args.batch_size=500
    kwargs = {'preload': True}
    train_loader, test_loader, args = load_dataset_exemplar(args,
                                                            shape=args.input_shape,
                                                            shuffle=False,
                                                            **kwargs)
    path_to_save_image = os.path.join(args.snap_dir, "generated_samples.npz")
    variation, exemplar = generate_samples_1d(test_loader, diffusion_model, AE, path_to_save_image)
    a=1


