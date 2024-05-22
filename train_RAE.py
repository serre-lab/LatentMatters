import torchvision
from models.AutoEncoder.one_d_AE import One_d_AE, RAE_28x28, RAE_48x48
from models.AutoEncoder.constrative_tools import ContrastiveLoss, barlow_loss, proto_net_loss, create_fake_label
import torch
from tqdm import tqdm
from torch.optim import Adam
#from torchvision.transforms import transforms
import torch.nn as nn
from utils.monitoring import plot_img, make_directories, make_grid, fig_to_img
import os
import argparse
from models.parser import dataset_args, training_args, ae_args, exp_args
from utils.dataloader import load_dataset_exemplar
from utils.custom_loader import Contrastive_augmentation, Contrastive_augmentation_fast, Cont_ProtAug
import matplotlib.pyplot as plt
import wandb
import numpy as np
import random

parser = argparse.ArgumentParser()
parser = dataset_args(parser)
parser = training_args(parser)
parser = ae_args(parser)
parser = exp_args(parser)
args = parser.parse_args()

wb_name = 'LDM_1D'
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

if args.model_name == 'RAE':
    AE = RAE_48x48(latent_size=args.latent_size,
               ae_dim=args.ae_dim,
               tanh=args.tanh,
               k=args.k,
               w_kl=args.w_kl,
               w_cons=args.w_cons,
               vq_reg=args.vq_reg,
               commit_vq=args.commit_vq,
               n_words=args.n_words,
               m=0.99).to(args.device)
else:
    raise NotImplementedError

#im = torch.randn(128, 1, 48, 48).to(args.device)
#out = AE(im)
#stop
args = make_directories(args)
if not args.debug:
    wandb.init(project=wb_name, config=vars(args), entity='victorboutin')
    wandb.run.name = args.model_signature
    wandb.run.save()
    torch.save(args, args.snap_dir + 'param.config')

param_to_optimize = list(AE.parameters())

if args.b_cons != 0:
    bar_loss = barlow_loss(args.latent_size).to(args.device)
    param_to_optimize += list(bar_loss.parameters())
    if args.pairing == 'augmentation':
        augment = Contrastive_augmentation(target_size=args.input_shape[1:], strength=args.contrastive_strength)
    elif args.pairing == 'prot_aug2':
        augment = Contrastive_augmentation(target_size=args.input_shape[1:], strength=args.contrastive_strength)

if args.proto_cons != 0:
    proto_distance = proto_net_loss(args.latent_size).to(args.device)
    loss_fn_proto = torch.nn.NLLLoss().to(args.device)
    param_to_optimize += list(proto_distance.parameters())

if args.w_cl != 0:
    linear_layer = nn.Linear(args.latent_size, 550).to(args.device)
    param_to_optimize += list(linear_layer.parameters())
    ce_loss = nn.CrossEntropyLoss()

print('# param : {0:,}'.format(sum(p.numel() for p in AE.parameters())))
#optimizer = Adam(AE.parameters(), lr=args.learning_rate, weight_decay=1e-5)
optimizer = Adam(param_to_optimize, lr=args.learning_rate, weight_decay=1e-5)
#distance = nn.MSELoss(reduction='mean')

if args.w_cons != 0:
    SimCLR_loss = ContrastiveLoss(args.batch_size).to(args.device)
    if args.pairing == 'augmentation':
        augment = Contrastive_augmentation(target_size=args.input_shape[1:], strength=args.contrastive_strength)
    elif args.pairing == 'prot_aug':
        augment = Cont_ProtAug(target_size=args.input_shape[1:], alpha=args.alpha_strength)
    elif args.pairing == 'prot_aug2':
        augment = Contrastive_augmentation(target_size=args.input_shape[1:], strength=args.contrastive_strength)
    #augment = Contrastive_augmentation_fast(train_loader, target_size=args.input_shape[1:], strength='normal')


for ep in range(args.epoch):
    train_stat = {'tot_loss': 0.0, 'reco_loss': 0.0, 'l1_loss': 0.0, 'l2_loss': 0.0, 'kl_loss': 0.0, 'cl_loss': 0.0,
                  'accu': 0.0, 'cons_loss': 0.0, 'proto_loss': 0.0, 'b_loss':0.0, 'vq_loss': 0.0}
    AE.train()
    if args.w_cl != 0:
        linear_layer.train()
    if args.b_cons != 0:
        bar_loss.train()
    if args.proto_cons != 0:
        proto_distance.train()
    l1_loss = torch.tensor([0.0]).to(args.device)
    l2_loss = torch.tensor([0.0]).to(args.device)
    kl_loss = torch.tensor([0.0]).to(args.device)
    cl_loss = torch.tensor([0.0]).to(args.device)
    vq_loss = torch.tensor([0.0]).to(args.device)
    cons_loss = torch.tensor([0.0]).to(args.device)
    proto_loss = torch.tensor([0.0]).to(args.device)
    b_loss = torch.tensor([0.0]).to(args.device)
    pbar = tqdm(train_loader)
    for idx_batch, (image, proto, label) in enumerate(pbar):
        image = image.to(args.device)
        label = label.to(args.device)
        optimizer.zero_grad()
        if args.w_cons != 0:
            if args.pairing == 'augmentation':
                image, image_aug = augment(image)
            elif args.pairing == 'prot_aug2':
                image, image_aug = augment(image, proto.to(args.device))
            elif args.pairing == 'prot_aug':
                image = augment(image)
        if args.b_cons != 0:
            if args.pairing == 'augmentation':
                image, image_aug = augment(image)
            elif args.pairing == 'prot_aug2':
                image, image_aug = augment(image, proto.to(args.device))
            elif args.pairing == 'prot_aug':
                image = augment(image)
        latent, decoded, loss_int = AE(image)
        if args.w_kl != 0:
            (q_mean, q_logvar, latent) = latent
            kl_loss = (-0.5 * torch.sum(1 + q_logvar - q_mean ** 2 - q_logvar.exp(), dim=1)).mean()
        if args.vq_reg is not None:
            vq_loss = loss_int
        loss_reco = ((image - decoded)**2).sum(dim=[1, 2, 3]).mean()
        if args.w_l1 != 0:
            l1_loss = latent.abs().sum(dim=1).mean()
        if args.w_l2 != 0:
            l2_loss = (latent**2).sum(dim=1).sqrt().mean()
        if args.w_cl != 0:
            logits = linear_layer(latent)
            cl_loss = ce_loss(logits, label)
        if args.proto_cons != 0:
            #labels_fake = create_fake_label(label)
            labels_fake = torch.arange(label.size(0)).long().to(args.device)
            latent_proto = AE.encode(proto.to(args.device))
            if args.w_kl != 0:
                latent_proto = latent_proto[-1]
            distance = proto_distance(latent, latent_proto)
            log_p_y = (-distance).log_softmax(dim=1)
            proto_loss = loss_fn_proto(log_p_y, labels_fake)
            #proto_loss = proto_distance(latent,latent_proto)
        if args.b_cons != 0:
            if args.pairing == 'augmentation' or (args.pairing == 'prot_aug2'):
                latent_2 = AE.encode(image_aug)
                if args.w_kl != 0:
                    latent_2 = latent_2[-1]
            else:
                raise NotImplementedError()
            b_loss = bar_loss(latent, latent_2)
        if args.w_cons != 0:
            with torch.no_grad():
                AE._momentum_update_key_encoder()
                if (args.pairing == 'augmentation') or (args.pairing == 'prot_aug2'):
                    latent_2 = AE.encoder_target(image_aug)
                else:
                    latent_2 = AE.encoder_target(proto.to(args.device))
                if args.w_kl != 0:
                    (q_mean_2, q_logvar_2) = latent_2
                    latent_2 = AE.reparametrize(q_mean_2, q_logvar_2)
            cons_loss = SimCLR_loss(latent, latent_2)


        loss = loss_reco + \
               args.w_l1 * l1_loss + \
               args.w_l2 * l2_loss + \
               args.w_kl * kl_loss + \
               args.w_cl * cl_loss + \
               args.w_vq * vq_loss + \
               args.w_cons * cons_loss + \
               args.b_cons * b_loss + \
               args.proto_cons * proto_loss
        train_stat['tot_loss'] += loss.item()
        train_stat['reco_loss'] += loss_reco.item()
        train_stat['l1_loss'] += l1_loss.item()
        train_stat['l2_loss'] += l2_loss.item()
        train_stat['kl_loss'] += kl_loss.item()
        train_stat['cl_loss'] += cl_loss.item()
        train_stat['cons_loss'] += cons_loss.item()
        train_stat['b_loss'] += b_loss.item()
        train_stat['proto_loss'] += proto_loss.item()
        train_stat['vq_loss'] += vq_loss.item()
        if args.w_cl != 0:
            _, max_idx = logits.max(dim=1)
            train_stat['accu'] += (max_idx == label).sum().item()/logits.size(0)
        if args.proto_cons != 0:
            y_pred = (-distance).softmax(dim=1)
            pred = y_pred.argmax(dim=1)
            train_stat['accu'] += (pred == labels_fake).sum().item()/y_pred.size(0)
        loss.backward()
        optimizer.step()
        pbar.set_description(
            f"{ep + 1}/{args.epoch}"
            f" -- rec: {loss_reco.item():.3f}"
            f" -- l1 : {l1_loss.item():.3f}"
            f" -- l2 : {l2_loss.item():.3f}"
            f" -- kl : {kl_loss.item():.3f}"
            f" -- cl : {cl_loss.item():.3f}"
            f" -- vq : {vq_loss.item():.3f}"
            f" -- cons : {cons_loss.item():.3f}"
            f" -- proto : {proto_loss.item():.3f}"
            f" -- barlow : {b_loss.item():.3f}"
        )

    train_stat['tot_loss'] /= idx_batch+1
    train_stat['reco_loss'] /= idx_batch + 1
    train_stat['l1_loss'] /= idx_batch + 1
    train_stat['l2_loss'] /= idx_batch + 1
    train_stat['kl_loss'] /= idx_batch + 1
    train_stat['cl_loss'] /= idx_batch + 1
    train_stat['vq_loss'] /= idx_batch + 1
    train_stat['cons_loss'] /= idx_batch + 1
    train_stat['proto_loss'] /= idx_batch + 1
    train_stat['b_loss'] /= idx_batch + 1
    train_stat['accu'] /= idx_batch + 1

    if not args.debug:
        wandb.log({"train/tot_loss": train_stat['tot_loss'],
                   "train/reco_loss": train_stat['reco_loss'],
                   "train/l2_loss": train_stat['l2_loss'],
                   "train/l1_loss": train_stat['l1_loss'],
                   "train/kl_loss": train_stat['kl_loss'],
                   "train/cl_loss": train_stat['cl_loss'],
                   "train/vq_loss": train_stat['vq_loss'],
                   "train/accu": train_stat['accu'],
                   "train/cons_loss": train_stat['cons_loss'],
                   "train/proto_loss": train_stat['proto_loss'],
                   "train/b_loss": train_stat['b_loss'],
                   "epochs": ep,
                   "lr": optimizer.param_groups[0]['lr']})

    if ep % 10 == 0:
        if not args.debug:
            tr_reco = wandb.Image(make_grid(decoded.cpu().detach(), ncol=10, nrow=11, normalize=True, scale_each=True),
                                caption='ep:{}'.format(ep))
            tr_im = wandb.Image(make_grid(image.cpu().detach(), ncol=10, nrow=11, normalize=True, scale_each=True),
                                caption='ep:{}'.format(ep))
            fig, ax = plt.subplots()
            ax.hist(torch.ravel(latent).cpu().detach().numpy(), bins=100, density=True)
            ax.set_xlim([-5, 5])
            ax.set_ylim([0, 1])

            wandb.log({"train/rec": tr_reco,
                       "train/im": tr_im,
                       "train/dis_latent": wandb.Image(fig)
                       }
                       )
            plt.close()
        with torch.no_grad():
            AE.eval()
            if args.b_cons != 0:
                bar_loss.eval()
            if args.proto_cons != 0:
                proto_distance.eval()
            pbar = tqdm(test_loader)
            l1_loss = torch.tensor([0.0]).to(args.device)
            l2_loss = torch.tensor([0.0]).to(args.device)
            kl_loss = torch.tensor([0.0]).to(args.device)
            vq_loss = torch.tensor([0.0]).to(args.device)
            cons_loss = torch.tensor([0.0]).to(args.device)
            proto_loss = torch.tensor([0.0]).to(args.device)
            b_loss = torch.tensor([0.0]).to(args.device)
            test_stat = {'tot_loss': 0.0, 'reco_loss': 0.0, 'l1_loss': 0.0, 'l2_loss': 0.0,
                         'kl_loss': 0.0, 'cons_loss': 0.0, 'proto_loss': 0.0, 'b_loss': 0.0,'vq_loss':0.0}
            for idx_batch, (image, proto, label) in enumerate(pbar):
                image = image.to(args.device)
                label = label.to(args.device)
                if args.w_cons != 0:
                    if args.pairing == 'augmentation':
                        image, image_aug = augment(image)
                    if args.pairing == 'prot_aug2':
                        image, image_aug = augment(image, proto.to(args.device))
                    elif args.pairing == 'prot_aug':
                        image = augment(image)
                if args.b_cons != 0:
                    if args.pairing == 'augmentation':
                        image, image_aug = augment(image)
                    elif args.pairing == 'prot_aug2':
                        image, image_aug = augment(image, proto.to(args.device))
                    elif args.pairing == 'prot_aug':
                        image = augment(image)
                latent, decoded, loss_int = AE(image)
                if args.w_kl != 0:
                    (q_mean, q_logvar, latent) = latent
                    kl_loss = (-0.5 * torch.sum(1 + q_logvar - q_mean ** 2 - q_logvar.exp(), dim=1)).mean()
                if args.vq_reg is not None:
                    vq_loss = loss_int
                loss_reco = ((image - decoded) ** 2).sum(dim=[1, 2, 3]).mean()
                if args.w_l1 != 0:
                    l1_loss = latent.abs().sum(dim=1).mean()
                if args.w_l2 != 0:
                    l2_loss = (latent ** 2).sum(dim=1).sqrt().mean()
                if args.proto_cons != 0:
                    #labels_fake = create_fake_label(label)
                    labels_fake = torch.arange(label.size(0)).long().to(args.device)
                    latent_proto = AE.encode(proto.to(args.device))
                    if args.w_kl != 0:
                        latent_proto = latent_proto[-1]
                    distance = proto_distance(latent, latent_proto)
                    log_p_y = (-distance).log_softmax(dim=1)
                    proto_loss = loss_fn_proto(log_p_y, labels_fake)
                    # proto_loss = proto_distance(latent,latent_proto)
                if args.w_cons != 0:
                    with torch.no_grad():
                        if (args.pairing == 'augmentation') or (args.pairing == 'prot_aug2'):
                            latent_2 = AE.encoder_target(image_aug)
                        else:
                            latent_2 = AE.encoder_target(proto.to(args.device))
                        if args.w_kl != 0:
                            (q_mean_2, q_logvar_2) = latent_2
                            latent_2 = AE.reparametrize(q_mean_2, q_logvar_2)
                    cons_loss = SimCLR_loss(latent, latent_2)
                if args.b_cons != 0:
                    if args.pairing == 'augmentation' or (args.pairing == 'prot_aug2'):
                        latent_2 = AE.encode(image_aug)
                        if args.w_kl != 0:
                            latent_2 = latent_2[-1]
                    else:
                        raise NotImplementedError()
                    b_loss = bar_loss(latent, latent_2)
                loss = loss_reco + \
                       args.w_l1 * l1_loss + \
                       args.w_l2 * l2_loss + \
                       args.w_kl * kl_loss + \
                       args.w_cl * kl_loss + \
                       args.w_vq * vq_loss + \
                       args.w_cons * cons_loss + \
                       args.b_cons * b_loss + \
                       args.proto_cons * proto_loss

                test_stat['tot_loss'] += loss.item()
                test_stat['reco_loss'] += loss_reco.item()
                test_stat['l1_loss'] += l1_loss.item()
                test_stat['l2_loss'] += l2_loss.item()
                test_stat['kl_loss'] += kl_loss.item()
                test_stat['cons_loss'] += cons_loss.item()
                test_stat['b_loss'] += b_loss.item()
                test_stat['vq_loss'] += vq_loss.item()
                test_stat['proto_loss'] += proto_loss.item()
                #test_stat['cl_loss'] += kl_loss.item()
            test_stat['tot_loss'] /= idx_batch+1
            test_stat['reco_loss'] /= idx_batch+1
            test_stat['l1_loss'] /= idx_batch+1
            test_stat['l2_loss'] /= idx_batch + 1
            test_stat['kl_loss'] /= idx_batch + 1
            test_stat['cons_loss'] /= idx_batch + 1
            test_stat['b_loss'] /= idx_batch + 1
            test_stat['vq_loss'] /= idx_batch + 1
            test_stat['proto_loss'] /= idx_batch + 1
            #test_stat['cl_loss'] /= idx_batch + 1
            if not args.debug:
                wandb.log({"test/tot_loss": test_stat['tot_loss'],
                           "test/reco_loss": test_stat['reco_loss'],
                           "test/l1_loss": test_stat['l1_loss'],
                           "test/l2_loss": test_stat['l2_loss'],
                           "test/kl_loss": test_stat['kl_loss'],
                           "test/cons_loss": test_stat['cons_loss'],
                           "test/b_loss": test_stat['b_loss'],
                           "test/vq_loss": test_stat['vq_loss'],
                           "test/proto_loss": test_stat['proto_loss'],
                           #"test/cl_loss": test_stat['cl_loss'],
                           })
                tr_reco = wandb.Image(make_grid(decoded.cpu(), ncol=10, nrow=11, normalize=True, scale_each=True),
                                      caption='ep:{}'.format(ep))
                tr_im = wandb.Image(make_grid(image.cpu(), ncol=10, nrow=11, normalize=True, scale_each=True),
                                    caption='ep:{}'.format(ep))
                fig, ax = plt.subplots()
                ax.hist(torch.ravel(latent).cpu().detach().numpy(), bins=100, density=True)
                ax.set_xlim([-5, 5])
                ax.set_ylim([0, 1])
                wandb.log({"test/rec": tr_reco,
                           "test/im": tr_im,
                           "test/dis_latent": wandb.Image(fig)})
                plt.close()

    if ep%(int(args.epoch/3)) == 0:
        if not args.debug:
            torch.save(AE.state_dict(), args.snap_dir + f'ep_{ep}.model')

if not args.debug:
    torch.save(AE.state_dict(), args.snap_dir + '_last.model')

