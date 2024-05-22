import argparse
from utils.monitoring import str2bool

def collect_as(coll_type):
  class Collect_as(argparse.Action):
    def __call__(self, parser, namespace, values, options_string=None):
      setattr(namespace, self.dest, coll_type(values))
  return Collect_as

def parse_args():
    # command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--double_z',           type=eval,      default=True, choices=[True, False])
    parser.add_argument('--z_channels',         type=int,       default=8, help='___')
    parser.add_argument('--resolution',         type=int,       default=48, help='___')
    parser.add_argument('--in_channels',        type=int,       default=1, help='___')
    parser.add_argument('--out_ch',             type=int,       default=1, help='___')
    parser.add_argument('--ch',                 type=int,       default=32, help='___')
    parser.add_argument('--num_res_blocks',     type=int,       default=2, help='___')
    parser.add_argument('--ch_mult',            nargs='+',      type=int,     default=[1, 2, 2, 4], help='___')
    parser.add_argument('--attn_resolutions',   nargs='+',      type=int,       default=[], help='___')
    parser.add_argument('--embed_dim',          type=int,       default=16, help='___')
    parser.add_argument('--n_embed',            type=int,       default=64, help='number of embeddings in dictionary for VQ-VAE')
    parser.add_argument('--dropout',            type=float,     default=0.0, help='___')
    parser.add_argument('--reg_weight',         type=float,     default=0.5, help='the weight on the kl and vq regularisation term')
    parser.add_argument('--p_weight',         type=float,     default=0.5, help='the weight on the perceptual regularisation term')
    parser.add_argument('--loss_name',           type=str, default=["LPIPS"], choices=["LPIPS","LPIPSDisc",'ContrastiveDisc'], help='type of the loss of the AE (using discriminative)')
    parser.add_argument('--disc_start', type=float, default=50001, help='when the disxcriminative loss starts')
    parser.add_argument('--disc_weight', type=float, default=0.5, help='weight of the discriminative component of the loss')
    parser.add_argument('--disc_num_layers', type=int, default=2, help='number of layer of the discriminator')
    return parser


def dataset_args(parser):
    parser.add_argument('--dataset', type=str, default='quickdraw_clust',
                        choices=['omniglot', 'quickdraw', 'quickdraw_clust'])
    parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
    parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
                        help='input batch size for training')
    parser.add_argument("--input_shape", nargs='+', type=int, default=(1, 48, 48), action=collect_as(tuple),
                        help='shape of the input [channel, height, width]')
    parser.add_argument('--input_type', type=str, default=None, choices=['binary'], help='type of the input')
    parser.add_argument('--preload', default=True, action='store_true', help='preload the dataset')
    parser.add_argument("--augment_class", action='store_true', default=False,
                        help="augment the number of classes by applying the same transform to examplar and variation")
    parser.add_argument("--transform_variation", action='store_true', default=False,
                        help="Increase the class originality by augmenting the variations (not the examplar)")
    parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                        metavar='EX_TYPE', help='type of exemplar')
    return parser


def training_args(parser):
    parser.add_argument('--epoch', type=int, default=100, metavar='EPOCH', help='number of epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate of the optimizer')
    parser.add_argument('--scheduler', type=str, default='none', help='learning rate scheduler',
                        choices=['none', 'one_cycle', 'step'])
    return parser


def ae_args(parser):
    parser.add_argument('--model_name', type=str, default='RAE', choices=['RAE'])
    parser.add_argument('--latent_size', type=int, default=128, help='number of latent dimension')
    parser.add_argument('--ae_dim', type=int, default=128, help='number of channel in the last conv layer of the AE')
    parser.add_argument('--tanh', type=str2bool, nargs='?', const=True, default=False, help="tanh on the latent space (default: False)")
    parser.add_argument('--k', type=int, default=0, help='number of active elements in the latent (default:None) ')
    parser.add_argument('--w_l1', type=float, default=0, help='l1 regularization on the latent')
    parser.add_argument('--w_l2', type=float, default=0, help='l2 regularization on the latent')
    parser.add_argument('--w_kl', type=float, default=0, help='kl regularization on the latent')
    parser.add_argument('--w_vq', type=float, default=0, help='vq regularization on the latent')
    parser.add_argument('--commit_vq', type=float, default=2.5, help='commitment weight on the vq loss')
    parser.add_argument('--n_words', type=int, default=512, help='number of words in the dico of the vqvae')
    parser.add_argument('--w_cl', type=float, default=0, help='classification regularization on the latent')
    parser.add_argument('--w_cons', type=float, default=0, help='contrastive regularization on the latent')
    parser.add_argument('--b_cons', type=float, default=0, help='barlow type of regularization')
    parser.add_argument('--proto_cons', type=float, default=0, help='prototype net regul')
    parser.add_argument('--alpha_strength', type=float, default=1, help='strenght of the augmentation (for proto_aug), should be below 10')
    parser.add_argument('--vq_reg',type=str, default=None, choices=[None, 'normal_vq', 'soft_vq'])
    parser.add_argument('--contrastive_strength', type=str, default='normal', choices=['normal', 'light', 'strong'],
                        help='decide the severeness of augmentation between 2 positive pairs in contrastive loss')
    parser.add_argument('--pairing', type=str, default='augmentation', choices=['augmentation', 'prototype', 'prot_aug', 'prot_aug2'],
                        help='paring mechanism between positive pairs (either based on augmentation or prototypes')
    return parser


def exp_args(parser):
    parser.add_argument('--tag', type=str, default='', help='tag of the experiment')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--seed', type=int, default=None, metavar='SEED', help='random seed (None is no seed)')
    parser.add_argument('--debug', default=False, action='store_true', help='debugging flag (do not save the network)')
    parser.add_argument('-od', '--out_dir', type=str, default='/media/data_cifs/projects/prj_ldm/exp/',
                        metavar='OUT_DIR', help='output directory for model snapshots etc.')
    parser.add_argument('--generate_img', default=False, action='store_true',
                        help='generate image at the end of the training')
    parser.add_argument('--exp_name', type=str, default="", help='name of the experiment')
    return parser

def diffusion_args(parser):
    parser.add_argument('--timestep', type=int, default=1000, metavar='TIMESTEP',
                        help='number of time step of the diffusion model')
    parser.add_argument("--betas", nargs='+', type=float, default=[0.0015, 0.0195],
                        help=' beta in the scheduler')
    parser.add_argument('--model_name', type=str, default='ldm_ddpm_stack',
                        choices=['ldm_ddpm_stack', 'ldm_ddpm', 'ldm_cfgdm_stack',
                                 'ldm_ddpm1d_stack', 'ldm_ddpm1d', 'ldm_cfgdm1d_stack'],
                        help="type of the diffusion model ['guided_diffus']")
    parser.add_argument('--unet_dim', type=int, default=64, metavar='UNET_DIM',
                        help='initial number of channels in Unet')
    parser.add_argument('--ae_path', type=str,
                        default='/media/data_cifs/projects/prj_ldm/exp/quickdraw_clust/autoencoder/KL/')
    parser.add_argument('--ae_name', type=str, default='', help='Name of autoencoder model')
    parser.add_argument("--unet_mult", nargs='+', type=float, default=[1, 2],
                        help='multiplier of the UNET')
    parser.add_argument('--attn_dim_head', type=int, default=32, metavar='ATTN_DIM_HEAD',
                        help='dimension of the attention head on the unet bottleneck')
    parser.add_argument('--attn_heads', type=int, default=4, metavar='ATTN_HEADS',
                        help='number of attention head on the unet bottlneck')
    parser.add_argument('--diffuser', type=str, default='conv1d_unet', choices=['conv1d_unet', 'att_mlp_1d','att_mlp_1d_light', 'att_mlp_1d_very_light','mlp_1d'],
                        help='architecture of the unet used in the diffusion model')
    parser.add_argument('--norm_type', type=str, default='rms', choices=['rms', 'groupnorm'],
                        help='Type of normalization layer before the attention layer')
    return parser