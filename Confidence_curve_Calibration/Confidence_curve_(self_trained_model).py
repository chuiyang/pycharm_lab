import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

bad_smiles = []
# bad_smiles = ['O=C1[C@H]2C[C@@H]1[C@@H]1C[C@H]2C1', 'O=C1O[C]2[C](OCO2)O1'
#               , 'C1[CH][C]2C[N][C]3N[C@@H]1[C@@H]23', ]
# bad_smiles = ['C']  # , 'CCC#N', 'COC'
# bad_smiles = ['C', 'C[C@@]12C[C@H]3[C@@H](O1)[C@H]3O2',
#               'C1C[C@@]23C[C@@H](C=CC2)[C@@H]13',
#               'C1[C@@H]2CC[C@H]3[C@H]4C[C@@]23[C@@H]14']

# original
# molecules = pd.read_csv('../../gpu_output/qm9_ens.csv', index_col=None).values
molecules = pd.read_csv('../prove_ale_epi_(knn)/knn_GCNN/knn_zinc_test/test_pred.csv', index_col=None).values
for i in bad_smiles:
    molecules = molecules[molecules[:, 0] != i]

n_mols = molecules.shape[0]
print(n_mols)
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


#  method
smiles = molecules[:, 0]
# true_file = pd.read_csv('../../gpu_output/fold_0/test_full.csv', index_col=None).values[:, :]
true_file = pd.read_csv('../prove_ale_epi_(knn)/knn_GCNN/knn_zinc_test/test_full.csv', index_col=None).values

for i in bad_smiles:
    true_file = true_file[true_file[:, 0] != i]
true_value = true_file[:, 1]
predicted_value = molecules[:, 1]
ale_uncertainty = molecules[:, 2]
epi_uncertainty = molecules[:, 3]
total_uncertainty = molecules[:, 2] + molecules[:, 3]

# total uncertainty
sort_file = np.vstack((smiles, true_value, predicted_value, total_uncertainty)).T
sort_file = sort_file[np.argsort(sort_file[:, 3])]
mae = []
for j in range(100):
    percent = 100 - j  # 100, 99, 98, 97.... 1
    amount = n_mols * (percent/100)
    y_true = sort_file[:int(amount), 1]
    y_pred = sort_file[:int(amount), 2]
    percent_mae = mean_absolute_error(y_true, y_pred)
    mae.append(percent_mae)
# oracle confidence curve
true_error = abs(true_value-predicted_value)
ora_sort_file = np.vstack((smiles, true_value, predicted_value, true_error)).T
ora_sort_file = ora_sort_file[np.argsort(ora_sort_file[:, 3])]
ora_mae = []
for j in range(100):
    percent = 100 - j  # 100, 99, 98, 97.... 1
    amount = n_mols * (percent/100)
    y_true = ora_sort_file[:int(amount), 1]
    y_pred = ora_sort_file[:int(amount), 2]
    percent_mae = mean_absolute_error(y_true, y_pred)
    ora_mae.append(percent_mae)
# epi
e = np.vstack((smiles, true_value, predicted_value, epi_uncertainty)).T
e = e[np.argsort(e[:, 3])]
ee = []
for j in range(100):
    percent = 100 - j  # 100, 99, 98, 97.... 1
    amount = n_mols * (percent/100)
    y_true = e[:int(amount), 1]
    y_pred = e[:int(amount), 2]
    percent_mae = mean_absolute_error(y_true, y_pred)
    ee.append(percent_mae)
# ale
a = np.vstack((smiles, true_value, predicted_value, ale_uncertainty)).T
a = a[np.argsort(a[:, 3])]
aa = []
for j in range(100):
    percent = 100 - j  # 100, 99, 98, 97.... 1
    amount = n_mols * (percent/100)
    y_true = a[:int(amount), 1]
    y_pred = a[:int(amount), 2]
    percent_mae = mean_absolute_error(y_true, y_pred)
    aa.append(percent_mae)
plt.cla()
plt.plot(range(100), ora_mae, c='gray', linestyle='--')
plt.plot(range(100), ee)
plt.plot(range(100), aa)
plt.plot(range(100), mae)


# name_ = []
# # for filename in os.listdir('../epi_ale_analysis/try/ensemble/'):
# for filename in os.listdir('../../cpu_output/try'):
#     # ecfp
#     ecfp_mae = []
#     ecfp_file = pd.read_csv(f'../../cpu_output/try/{filename}', index_col=None).values
#     # ecfp_file = pd.read_csv(f'../epi_ale_analysis/try/boot/{filename}', index_col=None).values
#     for i in bad_smiles:
#         ecfp_file = ecfp_file[ecfp_file[:, 0] != i]
#     print(ecfp_file.shape)
#     ecfp_epi_uncertainty = abs(ecfp_file[:, 2])
#     ecfp_sort_file = np.vstack((smiles, true_value, predicted_value, ecfp_epi_uncertainty)).T
#     ecfp_sort_file = ecfp_sort_file[np.argsort(ecfp_sort_file[:, 3])]
#
#     for j in range(100):
#         percent = 100 - j  # 100, 99, 98, 97.... 1
#         amount = n_mols * (percent/100)
#         y_true = ecfp_sort_file[:int(amount), 1]
#         y_pred = ecfp_sort_file[:int(amount), 2]
#         percent_mae = mean_absolute_error(y_true, y_pred)
#         ecfp_mae.append(percent_mae)
#
#     plt.plot(range(100), ecfp_mae)
#     name_.append(filename)

# plt.legend(['original', 'oracle', 'ecfp', 'fcfp'], loc='lower left')
# plt.legend(['oracle', 'epi', 'ale', 'total'] + name_, loc='lower left')
plt.legend(['oracle', 'epi', 'ale', 'total'], loc='lower left')

plt.grid()
# plt.ylim(0.0, 1.0)
# plt.xlim(0, 100)

plt.ylabel('MAE')
plt.xlabel(f'take away __% of molecule (out of {n_mols})')
plt.title(f'Confidence Curve_self_trained_model_ens')
plt.show()


