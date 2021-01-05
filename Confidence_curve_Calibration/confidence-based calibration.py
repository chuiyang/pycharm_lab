import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import os
# original
molecules = pd.read_csv('../qm9_complete_predictions.csv', index_col=None).values[:, :]
''' smiles	        0
dropout_predicted	1
dropout_ale_unc	    2
dropout_epi_unc	    3
TRUE	            4
ensemble_ale_unc	5
ensemble_epi_unc	6
ensemble_predicted	7
boot_ale_unc	    8
boot_epi_unc	    9
boot_predicted      10
'''

true = molecules[:, 4]
method = 2

if method == 1:
    name = 'MC-dropout'
    mean = molecules[:, 1]
    var_ale = molecules[:, 2]
    var_epi = molecules[:, 3]
    var_total = molecules[:, 2] + molecules[:, 3]
elif method == 2:
    name = 'Ensembling'
    mean = molecules[:, 7]
    var_ale = molecules[:, 5]
    var_epi = molecules[:, 6]
    var_total = molecules[:, 5] + molecules[:, 6]
else:
    name = 'Bootstrapping'
    mean = molecules[:, 10]
    var_ale = molecules[:, 8]
    var_epi = molecules[:, 9]
    var_total = molecules[:, 8] + molecules[:, 9]

# fcfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_mc_/model_save_epi_mc_ecfp_total_regu_0.0001/epi_output_ecfp.csv', index_col=None).values
# fcfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_/model_save_epi_ecfp_fcfp_total/epi_output_fcfp_regu.csv', index_col=None).values
# fcfp_file = pd.read_csv('../epi_ale_analysis/try/ensemble_epi_fcfp_mae.csv', index_col=None).values
# fcfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_boot_/model_save_epi_boot_ecfp_total_regu_0/epi_output_ecfp.csv', index_col=None).values
# var_fcfp_epi = fcfp_file[:, 2]
# ecfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_mc_/model_save_epi_mc_ecfp_patience100/epi_output_ecfp_0.csv', index_col=None).values
# ecfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_/model_save_epi_ecfp_fcfp_total/epi_output_ecfp_regu.csv', index_col=None).values
# ecfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_boot_/epi_output_ecfp_0.001.csv', index_col=None).values
# var_ecfp_epi = ecfp_file[:, 2]


for var in var_epi, var_ale, var_total:   # , var_ecfp_epi, var_fcfp_epi
    print(var.shape)
    cali = []
    num = []
    k_bin = np.linspace(0, 1, 20, endpoint=False)
    for j, con in enumerate(k_bin):
        count = 0
        for m, v, t in zip(mean, var, true):
            l_, u_ = scipy.stats.norm.interval(con, loc=m, scale=v**(1/2))
            if l_ < t < u_:
                count += 1
        num.append(count)
        cali.append(count/13083)
    plt.plot(k_bin, cali, linewidth=2)


# 確認自己訓練的model的pred和unc和之前拿到的是差不多的
# self pred new data
# new_file = pd.read_csv('../../gpu_output/qm9_ens_smiles0.csv', index_col=None).values
# new_pred = new_file[:, 1]
# new_a = new_file[:, 2]
# new_e = new_file[:, 3]
# new_t = new_file[:, 2] + new_file[:, 3]
# for var in new_e, new_a, new_t:   # , var_ecfp_epi, var_fcfp_epi
#     print(var.shape)
#     cali = []
#     num = []
#     k_bin = np.linspace(0, 1, 20, endpoint=False)
#     for j, con in enumerate(k_bin):
#         count = 0
#         for m, v, t in zip(new_pred, var, true):
#             l_, u_ = scipy.stats.norm.interval(con, loc=m, scale=v**(1/2))
#             if l_ < t < u_:
#                 count += 1
#         num.append(count)
#         cali.append(count/13083)
#     plt.plot(k_bin, cali, linewidth=2)


res_mae = []
res_file = pd.read_csv(f'../../cpu_output/try/epi_output_1.csv', index_col=None).values
# true_res = res_file[:, 1].astype(np.float)
pred_res = res_file[:, 2].astype(np.float)
true_res = pd.read_csv(f'../../cpu_output/try/epi_output_fcfp_regu.csv', index_col=None).values[:, 2]

for var in true_res, pred_res:
    print(var.shape)
    cali = []
    num = []
    k_bin = np.linspace(0, 1, 20, endpoint=False)
    for j, con in enumerate(k_bin):
        count = 0
        for m, v, t in zip(mean, var, true):
            l_, u_ = scipy.stats.norm.interval(con, loc=m, scale=v**(1/2))
            if l_ < t < u_:
                count += 1
        num.append(count)
        cali.append(count/13083)
    plt.plot(k_bin, cali, linewidth=2)


plt.legend(
    ['epi', 'ale', 'total'] +
    # ['new_e', 'new_a', 'new_t'] +
    ['pred_epi_ha', 'pred_epi_old'], loc='upper left')
plt.plot([0, 1], [0, 1], '--', c='gray')
plt.title(f'{name}')
plt.grid()
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.xlabel('confidence interval')
plt.ylabel('% of molecules fall in CI')
plt.show()






