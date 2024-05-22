# Latent Diffusion Matters: Human-like Sketches in One-shot Drawing Tasks

## 1. Download the database
In this article we have used the QuickDraw database (available [here](https://github.com/googlecreativelab/quickdraw-dataset)) and the Omniglot database (available [here](https://github.com/brendenlake/omniglot) )

## 2. Training RAEs
Once the databse are downloader you can train the RAE using the following command
```
python train_RAE.py --dataset_root YOUR_DATA_PATH --dataset quickdraw_clust --latent_size 128 --w_kl 0 --w_cl 0 --b_cons 1 --w_cons 0 --w_vq 0 --proto_cons 0 --device cuda:6 
```
In this script w_kl, w_vq, w_cl, proto_cons, w_cons, b_cons  correspond to the beta coefficient for the KL, VQ, classification, prototype-based, SimCLR and Barlow regularizers, respectively. 

## 3. Training Diffusion models
Once you have trained you RAEs, you can run the following script
  
```
python train_diffusion.py --dataset_root YOUR_DATA_PATH --dataset quickdraw_clust --ae_path YOUR_AE_PATH --ae_name YOUR_AE_NAME --timestep 1000 --diffuser att_mlp_1d --device cuda:6 
```