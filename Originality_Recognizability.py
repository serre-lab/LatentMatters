import torch
import argparse
import random
from RecOri.Ori_models import load_ori_critic_network
from RecOri.Reco_models import load_reco_critic_network, generate_evaluation_task
from utils.dataloader import load_dataset_exemplar
from RecOri.utils import load_ldm, generate_samples, append_OriReco_in_csv, load_ldm_1d, add_param_to_dico
import numpy as np
from utils.custom_transform import Scale_0_1_batch, Binarize_batch
import time
import json
from argparse import Namespace
import torchvision
import torch.nn as nn
import pandas as pd

import os
from utils.monitoring import plot_img


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',            type=str,       default='quickdraw_clust', choices=['omniglot','quickdraw', 'quickdraw_clust'])
parser.add_argument('--download_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--dataset_root', type=str, default="/media/data_cifs_lrs/projects/prj_control/data")
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 48, 48],
                    help='shape of the input [channel, height, width]')
parser.add_argument('--input_type', type=str, default='binary',
                    choices=['binary'], help='type of the input')
parser.add_argument('--n-train', default=1, type=int) #n shots
parser.add_argument('--n-test', default=1, type=int) #n shots
parser.add_argument('--k-train', default=60, type=int) #k ways
parser.add_argument('--k-test', default=20, type=int) #k ways
parser.add_argument('--q-train', default=5, type=int) #query
parser.add_argument('--q-test', default=1, type=int) #query
parser.add_argument("--exemplar_type", default='prototype', choices=['prototype', 'first', 'shuffle'],
                    metavar='EX_TYPE', help='type of exemplar')

args = parser.parse_args()
write_in_csv = True

if args.dataset == 'quickdraw_clust':
    nb_class = 115
    image_size = 48
    resize_func = nn.Identity()
    nb_bin = 10
    bin_size = 50
elif args.dataset == 'omniglot':
    nb_class = 150
    image_size = 50
    resize_func = torchvision.transforms.Resize(image_size)
    nb_bin = 10
    bin_size = 2
nb_test = 100 #100 #500
all_label = torch.arange(nb_class)
nb_block = 8

std_normalization = True
batch_stop = None
evaluation_task = generate_evaluation_task(batch_stop=batch_stop,
                                           nb_test=nb_test,
                                           nb_class=nb_class,
                                           nb_block=nb_block,
                                           k_test=args.k_test)

scale_01, binarize = Scale_0_1_batch(), Binarize_batch(binary_threshold=0.3)

path_to_csv = '/media/data_cifs/projects/prj_ldm/exp/quickdraw_clust/EXP/db_victor.csv'
#path_of_the_json = 'Exp/Exp_1_V.json'
path_of_the_json = 'Exp/Exp_128_V.json'
#path_of_the_json = 'Exp/Exp_LT_V.json'
exp_name = 'guided_bar_and_protonet_128'
saving_path = '/media/data_cifs/projects/prj_ldm/exp/quickdraw_clust/EXP/' + exp_name +'.pkl'

with open(path_of_the_json) as json_file:
    conf_exp = json.load(json_file)
dico_exp = conf_exp[exp_name]

if args.dataset == "quickdraw_clust":
    ori_model_name = "simclr_2022-09-26_10_05_52_z256_str=normal_QD"
    ori_path = "/media/data_cifs_lrs/projects/prj_ldm/critic_models/originality_models/"
    reco_model_name = "res_net_2022-09-26_10_02_17_z256_QD_NewResNEt_CropSeverAffine"
    reco_path = "/media/data_cifs_lrs/projects/prj_ldm/critic_models/recognizability_models/"
elif args.dataset == "omniglot":
    ori_model_name = "simclr_2022-02-22_14_21_52_z256_Omniglot_only"
    ori_path = "/media/data_cifs/projects/prj_zero_gene/neurips2022/embeddings"
    reco_model_name = "proto_net_2022-04-05_18_59_39_z256"
    reco_path = "/media/data_cifs/projects/prj_zero_gene/neurips2022/embeddings"

#if write_in_csv:
#    df = pd.read_csv(path_to_csv)

interesting_param_ae = ['w_kl', 'w_l1', 'k', 'tanh', 'w_l2', 'w_cl', 'w_cons', 'b_cons', 'proto_cons', 'w_vq', 'commit_vq',
                     'vq_reg', 'n_words', 'pairing', 'contrastive_strength', 'alpha_strength', 'model_name', 'latent_size', 'model_signature']
interesting_param_diff = ['diffuser', 'unet_mult', 'unet_dim', 'timestep', 'model_signature']


ori_critic_network = load_ori_critic_network(model_name=ori_model_name,
                                             path_to_embedding=ori_path,
                                             device=args.device,
                                             image_size=image_size)

reco_critic_network = load_reco_critic_network(classifier_name=reco_model_name,
                                               path_to_classifier=reco_path,
                                               image_size=image_size,
                                               device=args.device,
                                               args=args)

list_of_dico = []
with torch.no_grad():
    for each_conf in dico_exp:
        model_to_evaluate = each_conf["model_to_evaluate"]
        path_to_model_to_evaluate = each_conf["path_to_model"]
        autoencoder_name = each_conf["autoencoder_name"]
        path_to_autoencoder = each_conf["path_to_autoencoder"]
        if write_in_csv:
            df_csv = pd.read_csv(path_to_csv)
            if model_to_evaluate in df_csv['diff_model_signature'].values:
                print(f'{model_to_evaluate} already in database')
                continue
        path_diffusion = os.path.join(path_to_model_to_evaluate, model_to_evaluate)
        ae_path = os.path.join(path_to_autoencoder, autoencoder_name)
        if each_conf["model_type"] == "human":
            if args.dataset == 'omniglot':
                path_to_image = os.path.join(path_diffusion, "generated_img_all.npz")
            elif args.dataset == 'quickdraw_clust':
                path_to_image = os.path.join(path_diffusion, "test_img_prepro.npz")
            #path_to_image = os.path.join(path_diffusion, "human_training.pkl")
        else:
            path_to_image = os.path.join(path_diffusion, "generated_samples.npz")
        result_dico = {}

        ## Generate the samples, if not already done
        if os.path.exists(path_to_image):
            image_sample = np.load(path_to_image)
            if each_conf['model_type'] in ['ldm_ddpm1d_stack', 'ldm_cfgdm1d_stack']:
                param_diffusion, param_ae, _, _ = load_ldm_1d(path_to_diffusion=path_diffusion,
                                                              path_to_ae=ae_path,
                                                              device=args.device, only_param=True)
            if each_conf['model_type'] == "human":
                param_ae = Namespace(**{"model_signature": "human"})
                param_diffusion = Namespace(**{"model_signature": "human"})
            variation = image_sample['data']
            exemplar = image_sample['exemplar']

        else:
            args.batch_size = 500
            kwargs = {'preload': True}
            train_loader, test_loader, args = load_dataset_exemplar(args,
                                                                    shape=args.input_shape,
                                                                    shuffle=False,
                                                                    **kwargs)
            param_diffusion, param_ae, autoencoder, diffusion_model = load_ldm(path_to_diffusion=path_to_model,
                                                           path_to_ae=ae_path,
                                                           device=args.device)
            path_to_save_image = os.path.join(path_to_model, "generated_samples.npz")
            variation, exemplar = generate_samples(test_loader, diffusion_model, autoencoder, path_to_save_image)

        result_dico = add_param_to_dico(result_dico, param_ae, interesting_param_ae, 'ae')
        result_dico = add_param_to_dico(result_dico, param_diffusion, interesting_param_diff, 'diff')
        result_dico["path_to_model"] = path_diffusion
        result_dico["path_to_json"] = path_of_the_json
        result_dico["path_to_pkl"] = saving_path

        variation = torch.from_numpy(variation)
        exemplar = torch.from_numpy(exemplar)
        if args.dataset == "omniglot" and each_conf['model_type'] == "human":
            variation = variation[-150:]
            exemplar = exemplar[-150:]
        assert (variation.size(0) == nb_class and exemplar.size(0) == nb_class), \
            f"got a file with dim {variation.size(0)} but got {nb_class} classes"

        exemplar = exemplar.view(variation.size(0), 1, 48, 48)

        # Compute Originality
        all_cat, all_features, f_proto = [], [], []
        for cat_idx in range(variation.size(0)):
            one_cat, features_cat = [], []

            data = variation[cat_idx].to(args.device)
            proto = exemplar[cat_idx].unsqueeze(0).to(args.device)
            data = resize_func(data)
            proto = resize_func(proto)
            proto = binarize(scale_01(proto))
            data = binarize(scale_01(data))

            features_data = ori_critic_network(data)
            features_prototype = ori_critic_network(proto)
            # distance = (features_data.mean(dim=0, keepdim=True) - features_data).pow(2).sum(dim=1).sqrt()
            distance = (features_prototype - features_data).pow(2).sum(dim=1).sqrt()
            sorted_dist, sorted_idx = torch.sort(distance)
            for idx in range(nb_bin):
                selected_idx = sorted_idx[idx * bin_size: (idx + 1) * bin_size]
                features_cat.append(features_data[selected_idx])
                one_cat.append(data[selected_idx].cpu())
            features_cat = torch.stack(features_cat, dim=0)
            one_cat = torch.stack(one_cat, dim=0)
            all_features.append(features_cat)
            f_proto.append(features_prototype)
            all_cat.append(one_cat)

        all_cat = torch.stack(all_cat, dim=0)
        all_features = torch.stack(all_features, dim=0)

        f_proto = torch.stack(f_proto, dim=0)
        feature_size = all_features.size(-1)

        if std_normalization:
            normalizer = all_features.view(-1, feature_size).std(dim=-1).mean()
            all_features /= normalizer
            f_proto /= normalizer
        result_dico['originality'] = all_features.std(dim=-2, unbiased=True).mean(dim=-1).cpu()

        result_dico['features'] = all_features.cpu()
        result_dico['features_proto'] = f_proto.cpu()
        result_dico['originality_d'] = (all_features.view(nb_class, -1, 256) - f_proto).pow(2).mean(dim=2).sqrt().cpu()
        ## Compute Recognizability
        data_size = all_cat.size()
        target = torch.arange(args.k_test).to(args.device)
        result_dico["recognizability"] = []

        for idx_bin in range(nb_bin):
            tic = time.time()
            accu_tp = torch.zeros(len(all_label))
            data = all_cat[:, idx_bin]
            for idx_episode, episode in enumerate(evaluation_task):
                #if idx_episode % 350 == 0:
                #    print("{}/{}".format(idx_episode + 1, len(evaluation_task)))
                label = torch.tensor(episode).to(args.device)
                exemplar_se = exemplar[episode].to(args.device)
                data_ep = data[episode, torch.randint(data_size[2], (len(episode),)), :, :].to(args.device)
                exemplar_se = binarize(scale_01(exemplar_se))
                data_ep = binarize(scale_01(data_ep))
                y_pred, logits = reco_critic_network(exemplar_se, data_ep)
                pred = y_pred.argmax(dim=-1)
                n_b = pred.size(0)
                correct = pred.flatten().eq(target.repeat(n_b))
                correct_label = label[correct]
                for elem in correct_label:
                    accu_tp[elem] += 1
            tac = time.time()
            print(f'reco of bin{idx_bin+1} computed in {tac - tic:0.3f}s')
            result_dico['recognizability'].append(accu_tp / nb_test)
            result_dico['model_type'] = each_conf['model_type']
        result_dico['recognizability'] = torch.stack(result_dico['recognizability'], dim=1).cpu()
        print(f'model : {model_to_evaluate}')
        print(f'mean originality {result_dico["originality_d"].mean()}')
        print('mean recognizability', result_dico['recognizability'].mean())
        print('\n')
        list_of_dico.append(result_dico)
        if write_in_csv:
            append_OriReco_in_csv(result_dico, path_to_csv)

    if saving_path != '':
        torch.save(list_of_dico, saving_path)
    a=1