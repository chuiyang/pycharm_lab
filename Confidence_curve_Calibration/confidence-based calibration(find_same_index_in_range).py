
# 找original unc和新pred unc重疊的部分 (說明為什麼pred unc的calibration可以很好)

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
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
fcfp_file = pd.read_csv('../../cpu_output/try/error_output_1.csv').values
# fcfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_boot_/model_save_epi_boot_ecfp_total_regu_0/epi_output_ecfp.csv', index_col=None).values
var_fcfp_epi = fcfp_file[:, 2]
# ecfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_mc_/model_save_epi_mc_ecfp_patience100/epi_output_ecfp_0.csv', index_col=None).values
# ecfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_/model_save_epi_ecfp_fcfp_total/epi_output_ecfp_regu.csv', index_col=None).values
# ecfp_file = pd.read_csv('../epi_ale_analysis/model_save_epi_boot_/epi_output_ecfp_0.001.csv', index_col=None).values
# var_ecfp_epi = ecfp_file[:, 2]

# rmg_file = pd.read_csv('../rmg_analysis/model_save_epi_elbow/epi_output_total_1e-05.csv').values
# rmg_file = pd.read_csv('../rmg_analysis/model_save_epi_boot_elbow_2/epi_output_total_1e-05.csv').values
# var_rmg_epi = rmg_file[:, 2]


# for var in var_epi, var_ale, var_total, var_fcfp_epi:
#     print(var.shape)
plt.cla()
epi_dict = {}
fcfp_dict = {}
for var, add_dict in [var_epi, epi_dict], [var_fcfp_epi, fcfp_dict]:   # , var_ecfp_epi,  var_rmg_epi, var_ale, var_total,
    cali = []
    num = []
    k_bin = np.linspace(0, 1, 5, endpoint=False)
    for j, con in enumerate(k_bin):
        count = 0
        for i, m, v, t in zip(range(len(true)), mean, var, true):
            l_, u_ = scipy.stats.norm.interval(con, loc=m, scale=v**(1/2))
            if l_ < t < u_:
                count += 1
                add_dict.setdefault(j, set()).add(i)
        num.append(count)
        cali.append(count/13083)
    print(num)
    plt.plot(k_bin, cali, linewidth=2)

print(epi_dict)
print(fcfp_dict)
for i, j in epi_dict.items():
    print(i, len(j))
for i, j in fcfp_dict.items():
    print(i, len(j))

for key, i, j in zip(epi_dict.keys(), epi_dict.values(), fcfp_dict.values()):
    print(key, len(i & j))
plt.legend(['epi', 'fcfp_total'], loc='upper left')  # , 'ecfp_epi', 'rmg_epi' 'ale', 'total',
plt.plot([0, 1], [0, 1], '--', c='gray')
plt.title(f'{name}')
plt.grid()
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.xlabel('confidence interval')
plt.ylabel('% of molecules fall in CI')
plt.show()






