import os
import json
import torch

path_to_json = "/home/vboutin/prj_ldm/Exp/"
json_name = "Exp_128_V.json"
#json_name = "Exp_LT_V.json"
name_of_exp = 'guided_bar_and_protonet_128'
path_to_exp = "/media/data_cifs/projects/prj_ldm/exp/quickdraw_clust/EXP/EXP_BAR_AUG_AND_PROTONET_128/"
guided = True

if guided:
    folder_name = 'G_DIFF'
else:
    folder_name = 'DIFF'
all_json_path = os.path.join(path_to_json, json_name)
list_of_diffusion_model = next(os.walk(path_to_exp+folder_name))[1]
list_of_ae_name_ = next(os.walk(path_to_exp+'RAE'))[1]
list_of_ae_name = []
list_of_model_type = []

for each_diffusion_model in list_of_diffusion_model:
    path_to_config = os.path.join(path_to_exp+folder_name, each_diffusion_model)
    path_to_config += '/param.config'
    config_file = torch.load(path_to_config)
    assert config_file.ae_name in list_of_ae_name_, f"AE {config_file.ae_name} not in the folder"
    list_of_ae_name.append(config_file.ae_name)
    list_of_model_type.append(config_file.model_name)

with open(all_json_path, 'r+') as file:
    file_data = json.load(file)

    file_data[name_of_exp] = [{'model_type': list_of_model_type[idx],
                'model_to_evaluate': list_of_diffusion_model[idx],
                'path_to_model': path_to_exp+folder_name,
                'autoencoder_name': list_of_ae_name[idx],
                'path_to_autoencoder':path_to_exp+'RAE'}
               for idx in range(len(list_of_diffusion_model))]

    #file_data.append(dico_te)
    file.seek(0)
    json.dump(file_data, file, indent=4)
