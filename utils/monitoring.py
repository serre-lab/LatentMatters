import os
import datetime
import argparse
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'T', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'F', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_directories(args, model_class=None, ae_args = None):

    if args.model_name == 'autoencoder':
        full_model_name = args.regularisation
        if args.condition_type is not None:
            full_model_name += "_" + args.condition_type
    else:
        full_model_name = args.model_name

    if hasattr(args, 'attention_type'):
        full_model_name += '_' + args.attention_type
    args.model_signature = full_model_name
    if hasattr(args, 'ae_name'):
        if args.ae_name == '':
            args.model_signature += '_PIX'
    data_time = str(datetime.datetime.now())[0:19].replace(' ', '_')

    args.model_signature += '_' + data_time.replace(':', '_')
    if hasattr(args, 'diffuser'):
        args.model_signature += '_' + f'{args.diffuser}'
    if hasattr(args, 'latent_size'):
        nice_size = str(args.latent_size).replace("[", "").replace("]", "").replace(",", "x").replace(" ","")
        args.model_signature += '_' + f'lt={nice_size}'
    if hasattr(args, 'z_channels'):
        args.model_signature += '_' + f'z={args.z_channels}'
    if hasattr(args, 'reg_weight'):
        args.model_signature += '_' + f'wreg={args.reg_weight}'
    if hasattr(args, 'p_weight'):
        args.model_signature += '_' + f'wp={args.p_weight}'
    if hasattr(args, 'disc_weight'):
            args.model_signature += '_' + f'wd={args.disc_weight}'
    if hasattr(args, 'beta_vq'):
        args.model_signature += '_' + f'betaVQ={args.beta_vq}'
    if hasattr(args, 'num_attn'):
        args.model_signature += '_' + f'attn={args.num_attn}'
    if hasattr(args, 'unet_dim'):
        nice_size = str(args.unet_dim).replace("[", "").replace("]", "").replace(",", "-").replace(" ", "")
        args.model_signature += '_' + f'Unet_dim={nice_size}'
    if hasattr(args, 'w_l1'):
        if args.w_l1 != 0:
            args.model_signature += '_' + f'l1={args.w_l1}'
    if hasattr(args, 'w_l2'):
        if args.w_l2 != 0:
            args.model_signature += '_' + f'l2={args.w_l2}'
    if hasattr(args, 'w_kl'):
        if args.w_kl != 0:
            args.model_signature += '_' + f'kl={args.w_kl}'
    if hasattr(args, 'k'):
        if args.k != 0 :
            if args.k is not None:
                args.model_signature += '_' + f'k={args.k}'
    if hasattr(args, 'w_cl'):
        if args.w_cl != 0:
            args.model_signature += '_' + f'cl={args.w_cl}'
    if hasattr(args, 'proto_cons'):
        if args.proto_cons != 0:
            args.model_signature += '_' + f'prot={args.proto_cons}'

    if hasattr(args, 'b_cons'):
        if args.b_cons != 0:
            args.model_signature += '_' + f'b_cons={args.b_cons}_{args.contrastive_strength}'
            if args.pairing == 'prot_aug':
                args.model_signature += f'_{args.pairing}_str{args.alpha_strength}'
            else:
                args.model_signature += f'_{args.pairing}'
    if hasattr(args, 'w_cons'):
        if args.w_cons != 0:
            args.model_signature += '_' + f'cons={args.w_cons}_{args.contrastive_strength}'
            if args.pairing == 'prot_aug':
                args.model_signature += f'_{args.pairing}_str{args.alpha_strength}'
            else:
                args.model_signature += f'_{args.pairing}'
    if hasattr(args, 'vq_reg'):
        if args.vq_reg is not None:
            args.model_signature += '_' + f'{args.vq_reg}_{args.w_vq}_cvq={args.commit_vq}_nb_w={args.n_words}'
    if hasattr(args, 'unet_mult'):
        nice_size = str(args.unet_mult).replace("[", "").replace("]", "").replace(",", "-").replace(" ", "")
        args.model_signature += '_' + f'unet_mult={nice_size}'
    if ae_args is not None:
        if hasattr(ae_args, 'latent_size'):
            nice_size = str(ae_args.latent_size).replace("[", "").replace("]", "").replace(",", "x").replace(" ", "")
            args.model_signature += '_' + f'lt={nice_size}'
        if hasattr(ae_args, 'k'):
            if ae_args.k != 0:
                if ae_args.k is not None:
                    args.model_signature += '_' + f'k={ae_args.k}'
        if hasattr(ae_args, 'w_l1'):
            if ae_args.w_l1 != 0:
                args.model_signature += '_' + f'l1={ae_args.w_l1}'
        if hasattr(ae_args, 'w_l2'):
            if ae_args.w_l2 != 0:
                args.model_signature += '_' + f'l2={ae_args.w_l2}'
        if hasattr(ae_args, 'w_kl'):
            if ae_args.w_kl != 0:
                args.model_signature += '_' + f'kl={ae_args.w_kl}'
        if hasattr(ae_args, 'w_cl'):
            if ae_args.w_cl != 0:
                args.model_signature += '_' + f'cl={ae_args.w_cl}'
        if hasattr(ae_args, 'proto_cons'):
            if ae_args.proto_cons != 0:
                args.model_signature += '_' + f'prot={ae_args.proto_cons}'
        if hasattr(ae_args, 'vq_reg'):
            if ae_args.vq_reg is not None:
                args.model_signature += '_' + f'{ae_args.vq_reg}_{ae_args.w_vq}_cvq={ae_args.commit_vq}_nb_w={ae_args.n_words}'
        if hasattr(ae_args, 'w_cons'):
            if ae_args.w_cons != 0:
                args.model_signature += '_' + f'cons={ae_args.w_cons}_{ae_args.contrastive_strength}'
            if hasattr(ae_args, 'pairing'):
                if ae_args.pairing == 'prot_aug':
                    args.model_signature += f'_{ae_args.pairing}_str{ae_args.alpha_strength}'
                else:
                    args.model_signature += f'_{ae_args.pairing}'
        if hasattr(ae_args, 'b_cons'):
            if ae_args.b_cons != 0:
                args.model_signature += '_' + f'b_cons={ae_args.b_cons}_{ae_args.contrastive_strength}'
                if ae_args.pairing == 'prot_aug':
                    args.model_signature += f'_{ae_args.pairing}_str{ae_args.alpha_strength}'
                else:
                    args.model_signature += f'_{ae_args.pairing}'

    #if hasattr(args, 'exemplar'):
    #    if args.exemplar:
    #        args.model_signature += '_' + 'exVAE'
    #if hasattr(args, 'embed_dim'):
    #    args.model_signature += '_' + 'emb_dim{0}'.format(args.embed_dim)
    #if hasattr(args, 'z_channels'):
    #    args.model_signature += '_' + 'z_ch{0}'.format(args.z_channels)
    #if hasattr(args, 'z_size'):
    #    args.model_signature += '_' + 'z{0}'.format(args.z_size)
    #if hasattr(args, 'hidden_size') and hasattr(args, 'k'):
    #    args.model_signature += '_' + 'hid{0}_k{1}'.format(args.hidden_size, args.k)
    #if hasattr(args, 'hidden_prior') and hasattr(args, 'num_layer_prior'):
    #    args.model_signature += '_' + 'hid_p{0}_layer_p{1}'.format(args.hidden_prior, args.num_layer_prior)
    #if hasattr(args, 'time_step'):
    #    args.model_signature += '_' + 'T{0}'.format(args.time_step)
    #if hasattr(args, 'beta'):
    #    args.model_signature += '_' + 'beta{0}'.format(args.beta)
    #if hasattr(args, 'order'):
    #    args.model_signature += '_' + 'order{0}'.format(args.order)
    #if hasattr(args, 'size_factor'):
    #    args.model_signature += '_' + '{0}sf'.format(args.size_factor)
    #if hasattr(args, 'strength'):
    #    args.model_signature += '_' + 'str={0}'.format(args.strength)
    #if hasattr(args, 'annealing_time'):
    #    if args.annealing_time is not None:
    #        args.model_signature += '_' + 'BetaAnneal{}'.format(args.annealing_time)
    #if hasattr(args, 'shuffle_exemplar'):
    #    if args.shuffle_exemplar:
    #        args.model_signature += '_se'
    #if hasattr(args, 'rate_scheduler'):
    #    if args.rate_scheduler:
    #        args.model_signature += '_rc'
    #if hasattr(args, 'embedding_size'):
    #    args.model_signature += '_' + 'emb_sz={0}'.format(args.embedding_size)
    #if model_class == 'pixel_cnn' and hasattr(args, 'latent_size'):
    #    args.model_signature += '_latent_size=[{0},{1}]'.format(args.latent_size[0], args.latent_size[1])

    if args.tag != '':
        args.model_signature += '_' + args.tag

    if args.exp_name != "":
        args.model_signature = args.exp_name + "_" + args.model_signature
    if model_class is None:
        if args.exp_name != "":
            snapshots_path = os.path.join(args.out_dir, args.dataset, "EXP", args.exp_name, full_model_name)
        else:
            snapshots_path = os.path.join(args.out_dir, args.dataset, full_model_name)
    else:
        snapshots_path = os.path.join(args.out_dir, args.dataset, model_class)
    if args.model_name == "autoencoder":
        snapshots_path = os.path.join(args.out_dir, args.dataset, args.model_name, full_model_name)
    #if 'ldm' in args.model_name:
    #    snapshots_path = os.path.join(args.out_dir, args.dataset, 'ldm', full_model_name)
    #if 'ldm' in args.model_name:
    #    snapshots_path = os.path.join(args.out_dir, args.dataset, args.model_name, full_model_name)

    args.snap_dir = snapshots_path + '/' + args.model_signature + '/'

    #if args.model_name in ['ns', 'tns', 'hfsgm']:
    #    args.snap_dir = snapshots_path + '/' + args.model_signature + '_' + str(args.c_dim) + '_' + str(
    #        args.z_dim) + '_' + str(args.hidden_dim)
    #    if args.model_name == 'tns':
    #        args.snap_dir += '_' + str(args.n_head)
    #    args.snap_dir += '/'

    if not args.debug:
        os.makedirs(snapshots_path, exist_ok=True)
        os.makedirs(args.snap_dir, exist_ok=True)
        args.fig_dir = args.snap_dir + 'fig/'
        os.makedirs(args.fig_dir, exist_ok=True)
    else:
        args.fig_dir = None
    return args


def show(img, title=None, saving_path=None, figsize=(8, 8), dpi=100, cmap=None, axs=None):
    npimg = img.numpy()
    if axs is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest', cmap=cmap)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        if saving_path is None :
            plt.show()
        else:
            plt.savefig(saving_path + '/' + title + '.png')
        plt.close()
    else :
        axs.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest', cmap=cmap)


def plot_img(data, nrow=4, ncol=8, padding=2, normalize=True, saving_path=None, title=None, pad_value=0, figsize=(8, 8), dpi=100, scale_each=False, cmap=None, axs=None):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    show(data_to_plot.detach().cpu(), saving_path=saving_path, title=title, figsize=figsize, dpi=dpi, cmap=cmap, axs=axs)


def make_grid(data, nrow=4, ncol=8, padding=2, normalize=True, pad_value=0, scale_each=False):
    nb_image = nrow * ncol
    data_to_plot = torchvision.utils.make_grid(data[:nb_image], nrow=ncol, padding=padding, normalize=normalize,
                                               pad_value=pad_value, scale_each=scale_each)
    return data_to_plot

def fig_to_img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img



