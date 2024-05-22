import os
import torch
from models.AutoEncoder.Autoencoders import AutoencoderKL, VQModel
from models.Diffusion.OpenAIUnet import UNetModel
from models.Diffusion.sampler import DDPM_ELBO
from utils.monitoring import plot_img
import numpy as np
import pandas as pd
import math

def load_weights(out_dir, model_name=None, mode='end', weight=True):

    if model_name is not None:
        path_to_load = os.path.join(out_dir, model_name)
    else:
        path_to_load = out_dir
    path_args = os.path.join(path_to_load, 'param.config')
    loaded_args = torch.load(path_args)
    if mode == 'end':
        path_weights = os.path.join(path_to_load, '_end.model')
    elif mode == 'best':
        path_weights = os.path.join(path_to_load, '_best.model')

    if weight:
        loaded_weights = torch.load(path_weights, map_location='cpu')
    else:
        loaded_weights = None
    return loaded_args, loaded_weights

def load_ae(path_to_ae, device='cpu', only_param=False):
    path_to_ae_param =  os.path.join(path_to_ae, 'param.config')
    path_to_ae_weight = os.path.join(path_to_ae, 'last.ckpt')
    param_ae = torch.load(path_to_ae_param)
    if param_ae.regularisation == 'KL':
        autoencoder = AutoencoderKL.load_from_checkpoint(path_to_ae + "/last.ckpt")
    elif param_ae.regularisation == 'VQ':
        autoencoder = VQModel.load_from_checkpoint(path_to_ae + "/last.ckpt")
    autoencoder = autoencoder.to(device)
    autoencoder.freeze()
    return param_ae, autoencoder

def load_ldm_1d(path_to_diffusion, path_to_ae, device='cpu', only_param=False):
    path_to_diffusion_param = os.path.join(path_to_diffusion, 'param.config')
    param_diffusion = torch.load(path_to_diffusion_param)

    path_to_ae_param = os.path.join(path_to_ae, 'param.config')
    param_ae = torch.load(path_to_ae_param)

    assert param_ae.model_signature == param_diffusion.ae_name, 'The AE name do not coresponds to the one the diffusion model has been trained one'
    if only_param:
        return param_diffusion, param_ae, None, None
    else:
        raise NotImplementedError()


def load_ldm(path_to_diffusion, path_to_ae, device='cpu', only_param=False):
    #Load the diffusion model
    path_to_diffusion_param = os.path.join(path_to_diffusion, 'param.config')
    param_diffusion = torch.load(path_to_diffusion_param)

    path_to_ae_param = os.path.join(path_to_ae, 'param.config')
    param_ae = torch.load(path_to_ae_param)

    assert param_ae.model_signature == param_diffusion.autoenc_name, 'The AE name do not coresponds to the one the diffusion model has been trained one'

    if only_param:
        return param_diffusion, param_ae, None, None

    if not only_param:
        path_to_diffusion_weight = os.path.join(path_to_diffusion, '_end.model')

        path_to_ae_weight = os.path.join(path_to_ae, 'last.ckpt')


    if param_ae.regularisation == 'KL':
        autoencoder = AutoencoderKL.load_from_checkpoint(path_to_ae + "/last.ckpt")
    elif param_ae.regularisation == 'VQ':
        autoencoder = VQModel.load_from_checkpoint(path_to_ae + "/last.ckpt")
    autoencoder = autoencoder.to(device)
    autoencoder.freeze()

    if param_diffusion.model_name == 'ldm_ddpm_stack':
        conditionning = 'stack'  # stack the input to conditionning

    if param_diffusion.arch_unet == 'openai':
        attn_res = []
        ds = 1
        for i in range(len(param_diffusion.unet_dim)):
            attn_res.append(ds)
            ds = 2 * ds
        attn_res = tuple(attn_res)
        unet_model = UNetModel(
            image_size=autoencoder.latent_size[-1],
            in_channels=autoencoder.latent_size[0],
            out_channels=autoencoder.latent_size[0],
            model_channels=param_diffusion.n_feat,  ##224
            num_res_blocks=2,
            attention_resolutions=attn_res,  # [2]
            channel_mult=param_diffusion.unet_dim,  # [1, 4]
            num_head_channels=32,
            conditioning=conditionning
        ).to(device)
        diffusion_model = DDPM_ELBO(nn_model=unet_model,
                                    betas=param_diffusion.betas,
                                    n_T=param_diffusion.timestep,
                                    device=device,
                                    drop_prob=param_diffusion.drop_out).to(device)
        diffusion_model.load_state_dict(torch.load(path_to_diffusion_weight))
        diffusion_model.eval()
    else:
        raise NotImplementedError()

    return param_diffusion, param_ae, autoencoder, diffusion_model

def generate_samples(test_dataloader, diffusion_model, autoencoder, path_to_save_image, deterministic=False):
    all_generation = []
    all_exemplar = []
    for idx_batch, (image, exemplar, label) in enumerate(test_dataloader):
        exemplar = exemplar.to(diffusion_model.device)
        if autoencoder.condition_type == "EncDec":
            condition = exemplar
        else:
            condition = None
        if idx_batch == 0:
            scale_factor = autoencoder.get_scale_factor(exemplar, condition=condition)
        exemplar_latent = autoencoder.get_latent(exemplar, condition=condition, deterministic=deterministic)
        sampled_latent_tr = diffusion_model.sample_c(image_size=exemplar_latent.size()[1:],
                                                     batch_size=exemplar.size(0),
                                                     cond=exemplar_latent)
        if autoencoder.condition_type is not None:
            condition = exemplar
        else:
            condition = None

        if isinstance(autoencoder, AutoencoderKL):
            sample_images_tr = autoencoder.decode(1 / scale_factor * sampled_latent_tr, condition=condition)
        elif isinstance(autoencoder, VQModel):
            quantized_latent_tr, _, _ = autoencoder.quantize(sampled_latent_tr)
            sample_images_tr = autoencoder.decode(quantized_latent_tr, condition=condition)
        all_generation.append(sample_images_tr.cpu().numpy())
        all_exemplar.append(exemplar[0:1].cpu().numpy())
        print(f'{idx_batch+1}/{len(test_dataloader)}')
    all_generation = np.stack(all_generation, 0)
    all_exemplar = np.stack(all_exemplar, 0)
    np.savez_compressed(path_to_save_image, data=all_generation, exemplar=all_exemplar)
    print('generation has been saved')
    return all_generation, all_exemplar

def generate_samples_1d(test_dataloader, diffusion_model, autoencoder, path_to_save_image=None, w_guidance=1, deterministic=False):
    with torch.no_grad():
        all_generation = []
        all_exemplar = []
        for idx_batch, (image, exemplar, label) in enumerate(test_dataloader):
            exemplar = exemplar.to(diffusion_model.device)

            if idx_batch == 0:
                image = image.to(diffusion_model.device)
                mean, std = autoencoder.estimate_stats(image)
            lt_proto = autoencoder.get_latent_from_img(exemplar, mean=mean, std=std)
            sample_te = diffusion_model.sample_c(image_size=lt_proto.size()[1:],
                                                 batch_size=exemplar.size(0),
                                                 w_guidance=w_guidance,
                                                 cond=lt_proto)
            sample_te = autoencoder.get_img_from_latent(sample_te, mean=mean, std=std)
            all_generation.append(sample_te.cpu().detach().numpy())
            all_exemplar.append(exemplar[0:1].cpu().detach().numpy())
            print(f'{idx_batch + 1}/{len(test_dataloader)}')
        all_generation = np.stack(all_generation, 0)
        all_exemplar = np.stack(all_exemplar, 0)
        if path_to_save_image is not None:
            np.savez_compressed(path_to_save_image, data=all_generation, exemplar=all_exemplar)
            print('generation has been saved')
    return all_generation, all_exemplar

def append_OriReco_in_csv(result_dico, path_csv):
    data = {}
    for each_elem in result_dico.keys():
        if each_elem not in ['features', 'features_proto']:
            if each_elem in ['originality', 'recognizability', 'originality_d']:
                to_add = round(result_dico[each_elem].mean().item(), 3),
            elif each_elem in ['latent_size']:
                to_add = str(result_dico[each_elem])
            else:
                to_add = result_dico[each_elem]
            data[each_elem]=to_add

    #data = {
    #    'model type': result_dico['model type'],
    #    "ae name": result_dico["ae_name"],
    #    "diffusion name": result_dico["diffusion_name"],
    #    "regularization AE": result_dico["regularization"],
    #    "size latent": str(result_dico["size_latent"]),
    #    "mean originality": round(result_dico["originality"].mean().item(), 3),
    #    "mean recognizability": round(result_dico['recognizability'].mean().item(), 3),
    #    "path to model": result_dico["path_to_model"],
    #    "ae_w_reg": round(result_dico["w_reg"], 2),
    #    "ae_w_disc": round(result_dico["w_disc"], 2),
    #    "ae_w_perc": round(result_dico["w_perc"], 2),
    #    "json path": result_dico["path_to_json"],
    #    "n_embed": result_dico["n_embed"],
    #}


    df = pd.DataFrame(data, index=[0])

    # if file does not exist write header
    if not os.path.isfile(path_csv):
        df.to_csv(path_csv, index=False, header=True)
    else:  # else it exists so append without writing the header
        df_csv = pd.read_csv(path_csv)
        if (df['ae_model_signature'].item() in list(df_csv['ae_model_signature'])) and (df['diff_model_signature'].item() in list(df_csv['diff_model_signature'])):
            print(f'{df["diff_model_signature"].item()} already there')
        else:
            df.to_csv(path_csv, mode='a', index=False, header=False)

def add_param_to_dico(result_dico, param, interesting_param, type_model):
    if param is not None:
        param_dico = vars(param)
    else:
        param_dico = {}
    for each_param in interesting_param:
        name = each_param
        if each_param in param_dico.keys():
            if type(param_dico[each_param]) == bool:
                to_save = str(param_dico[each_param])
            elif type(param_dico[each_param]) == list:
                to_save = str(param_dico[each_param])
            elif each_param == 'model_signature':
                to_save = str(param_dico[each_param])
                name = type_model + '_' + each_param
            else:
                to_save = param_dico[each_param]
                name = each_param
            result_dico[name] = to_save
        else:
            result_dico[name] = 0
    return result_dico