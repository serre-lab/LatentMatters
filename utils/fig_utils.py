from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.parametric_fit import parametric_fit
from numpy.polynomial import polynomial as pl
from scipy import interpolate
import math
def ori_reco_with_fit(df_f, param, weight_extremum=None, polydeg=2, s=0, log10=False, normalizer=1, down_weight=None):


    all_reg, all_ori, all_accu = [], [], []
    for i, (index, row) in enumerate(df_f.iterrows()):
        accu = row['recognizability']
        ori = row['originality_d']*normalizer
        w = row[param]
        if log10:
            all_reg.append(math.log10(w))
        else:
            all_reg.append(w)
        all_accu.append(accu)
        all_ori.append(ori)
    reg_np = torch.tensor(all_reg)
    #if log10:
    #    reg_np = np.log10(reg_np)
    ori_np = torch.tensor(all_ori)
    accu_np = torch.tensor(all_accu)
    P = torch.stack([reg_np, ori_np, accu_np], dim=1)
    coeff_reg, coeff_ori, coeff_accu = parametric_fit(P, analyze=False, max_iter=20, polydeg=polydeg, title='test', weight_extremum=weight_extremum, down_weight=down_weight)
    xx = torch.linspace(0, 1, 100)
    fit_reg = pl.polyval(xx, coeff_reg)
    fit_ori = pl.polyval(xx, coeff_ori)
    fit_accu = pl.polyval(xx, coeff_accu)

    param_fit_spline = np.linspace(min(all_reg), max(all_reg), 100)
    tck_ori = interpolate.splrep(all_reg, all_ori, s=s, k=4)
    ori_fit_spline = interpolate.splev(param_fit_spline, tck_ori)

    tck_reco = interpolate.splrep(all_reg, all_accu, s=s, k=4)
    reco_fit_spline = interpolate.splev(param_fit_spline, tck_reco)

    #tck_reco = interpolate.splrep(all_reg, all_accu)
    #reco_fit_spline = interpolate.splev(np.arange(min(all_reg), max(all_reg), 100), tck_reco)
    return all_reg, all_ori, all_accu, fit_reg, fit_ori, fit_accu, param_fit_spline, ori_fit_spline, reco_fit_spline