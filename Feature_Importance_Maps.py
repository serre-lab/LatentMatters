import torch
import torch.nn as nn
import os
import argparse
from models.AutoEncoder.one_d_AE import RAE_48x48
from models.Diffusion.one_d_diffusion import Unet1D_MLP
from models.Diffusion.one_d_diffuser import DDPM_ELBO
from utils.custom_transform import Scale_0_1_batch, Binarize_batch
import numpy as np
from utils.monitoring import plot_img
import random
from utils.monitoring import plot_img
from matplotlib import pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='/media/data_cifs_lrs/projects/prj_ldm/exp/quickdraw_clust/EXP/EXP_CL_128/',
                    help='path to auto-encoder')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='device')
args = parser.parse_args()


BASE_PATH_DIFF = args.base_path + "/G_DIFF/"
BASE_PATH_RAE = args.base_path + "/RAE/"

list_of_model = next(os.walk(BASE_PATH_DIFF))[1]

SEED = 59

##load the prototype
data_path = '/media/data_cifs/projects/prj_control/data/quick_draw/qd_ClickMe.npz'
scale_01, binarize = Scale_0_1_batch(), Binarize_batch(binary_threshold=0.3)

db = np.load(data_path)
proto = torch.from_numpy(db['prototype'])
cat_name = db['category_name']
proto = binarize(scale_01(proto))
selected_proto = [0, 6, 9, 12, 14, 27, 29, 33, 35, 45, 46, 50, 51, 55, 70, 68, 69, 82, 83,
                  86, 92, 94, 96, 99, 42]

#selected_proto = np.arange(100)
new_cat_name = []
for each_cat in selected_proto:
    new_cat_name.append(cat_name[each_cat])
proto = proto[selected_proto]
proto_batch = proto.unsqueeze(dim=1).repeat(1, len(selected_proto), 1, 1, 1)



for each_model_name in list_of_model:
    ## load the diffsuion model
    path_to_diff = os.path.join(BASE_PATH_DIFF, each_model_name)
    path_to_weight_diff = path_to_diff + "/_last.model"
    path_to_args_diff = path_to_diff + "/param.config"
    weight_diff = torch.load(path_to_weight_diff)
    args_diff = torch.load(path_to_args_diff)

    if args_diff.model_name == 'ldm_cfgdm1d_stack':
        drop_out = 0.1
        conditioning = "stack"

    ## load the corresponding AE
    path_to_rae = os.path.join(BASE_PATH_RAE, args_diff.ae_name)
    path_to_weight_ae = path_to_rae + "/_last.model"
    path_to_args_ae = path_to_rae + "/param.config"
    weight_ae = torch.load(path_to_weight_ae)
    args_ae = torch.load(path_to_args_ae)

    denoiser_model = Unet1D_MLP(dim=args_diff.unet_dim,
                                latent_size=args_ae.latent_size,
                                dim_mults=args_diff.unet_mult,
                                channels=args_diff.input_shape[0],
                                attn_dim_head=args_diff.attn_dim_head,
                                attn_heads=args_diff.attn_heads,
                                conditioning=conditioning,
                                resnet_block_groups=8,
                                norm_type=args_diff.norm_type,
                                ae_enc=None
                                )

    diffusion_model = DDPM_ELBO(nn_model=denoiser_model,
                                betas=(args_diff.betas[0], args_diff.betas[1]),
                                n_T=args_diff.timestep,
                                device=args.device,
                                drop_prob=drop_out)

    diffusion_model.load_state_dict(weight_diff)
    diffusion_model.to(args.device)
    diffusion_model.eval()

    AE = RAE_48x48(**vars(args_ae))
    AE.load_state_dict(weight_ae, strict=False)
    AE.to(args.device)
    AE.eval()

    mean, std = AE.estimate_stats(proto_batch.view(-1, 1, 48, 48).to(args.device))

    #multiple_coeff = torch.sqrt(1 - diffusion_model.alphas_cumprod)

    def decode(z):
        return AE.get_img_from_latent(z, mean=mean, std=std)

    def batch_jacobian(func, x, create_graph=False):
      # x in shape (Batch, Length)
      def _func_sum(x):
        return func(x).sum(dim=0)
      return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph, vectorize=False).permute(3, 0, 1, 2,4)


    sampled_ts = [torch.linspace(100, 500, 20).long(), torch.linspace(510, 800, 30).long(),
                  torch.linspace(801, 999, 50).long()]
    sampled_ts = torch.cat(sampled_ts, dim=0)
    mult = torch.sqrt(1 - diffusion_model.alphas_cumprod)
    mult_f = mult[sampled_ts]

    with torch.no_grad():
        all_conditional_score = []
        all_unconditional_score = []
        all_image = []
        all_phi = []
        for idx_category in range(proto.size(0)):
            tic = time.time()
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            print("{}/{}".format(idx_category + 1, proto.size(0)))
            exemplar = proto_batch[idx_category].to(args.device)
            exemplar_lt = AE.get_latent_from_img(exemplar, mean=mean, std=std)
            data = torch.zeros_like(exemplar)
            #print(exemplar_lt)
            image, cond_score, uncond_score, all_img = diffusion_model.compute_guidance_score(
                image_size=(args_ae.latent_size,),
                batch_size=exemplar.size(0),
                cond=exemplar_lt)

            all_img_f = all_img[sampled_ts]
            cond_score_f = cond_score[sampled_ts]
            uncond_score_f = uncond_score[sampled_ts]
            original_size = all_img_f.size()
            jacob_img = batch_jacobian(decode, all_img_f.view(-1, 128))
            jacob_img = jacob_img.view(*original_size[0:2], *jacob_img.size()[-4:])

            dx = mult_f.unsqueeze(-1).unsqueeze(-1) * uncond_score_f
            dy = mult_f.unsqueeze(-1).unsqueeze(-1) * cond_score_f
            score = 2 * dy - dx
            prj_score = torch.einsum('i b c w h f, i b f -> i b c w h', jacob_img, score)
            phi = prj_score.abs().sum(dim=0)
            all_phi.append(phi)
            del prj_score
            del jacob_img
            del phi
            del uncond_score
            del cond_score
            torch.cuda.empty_cache()
            duration = time.time()-tic
            print(f"duration for one cat {duration:0.2f}")

        all_phi = torch.stack(all_phi, dim=0)
    np.savez_compressed(path_to_diff+'/diff_attri2.npz', attribution_map=all_phi.cpu().numpy())

    print(f"time to run the method {duration:0.2f}")


