import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

# bad_smiles = ['C', 'CC1CCC1', 'C1[C@H]2O[C@@H]1[C@@H]1C[C@H]2O1', 'C1[C@H]2C[C@H]3O[C@H]3C[C@@H]1O2', 'O=CC#C'
#    , 'OCCC(O)(C#N)C#N', 'C[C@@]1(O)[C@H]2CC=C[C@@H]1O2', 'CC1CC1', 'N#C[C@]12C[C@H]1N1C[C@@H]2C1', 'CNC(=O)NC(=O)C'
#    , 'C1N2C[C@]31CO[C@H]1[C@@H]2[C@@H]31'
#    , 'C1[C@H]2[C@@H]1[C@H]1C[C@H]3C[C@@H]2[C@@H]13', 'C1C=CC=CC=C1', 'C1[C@H]2C[C@@H]3O[C@H]2C[C@H]13'
#    , '[NH]C1=CC=C(C=NO1)[NH]', 'C1CCCCCC1', 'CC1CC1']
# bad_smiles = ['C', 'O=CC#C', 'CCC#N', 'COC', 'CC1CC1']  # ensemble
# bad_smiles = ['C1COC1', 'C', 'CC1CC1', 'C1CCCCCC1', 'CC1CCC1', 'COC']  # boot
bad_smiles = []

# original
molecules = pd.read_csv('../qm9_complete_predictions.csv', index_col=None).values
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
true_value = molecules[:, 4]
method = 2

if method == 1:
    name = 'MC-dropout'
    predicted_value = molecules[:, 1]
    ale_uncertainty = molecules[:, 2]
    epi_uncertainty = molecules[:, 3]
    total_uncertainty = molecules[:, 2] + molecules[:, 3]
elif method == 2:
    name = 'Ensembling'
    predicted_value = molecules[:, 7]
    ale_uncertainty = molecules[:, 5]
    epi_uncertainty = molecules[:, 6]
    total_uncertainty = molecules[:, 5] + molecules[:, 6]
else:
    name = 'Bootstrapping'
    predicted_value = molecules[:, 10]
    ale_uncertainty = molecules[:, 8]
    epi_uncertainty = molecules[:, 9]
    total_uncertainty = molecules[:, 8] + molecules[:, 9]

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


name_ = []
# for filename in os.listdir('../epi_ale_analysis/try/ensemble/'):
for filename in os.listdir('../../cpu_output/try'):
    # ecfp
    ecfp_mae = []
    ecfp_file = pd.read_csv(f'../../cpu_output/try/{filename}', index_col=None).values

    # ecfp_file = pd.read_csv(f'../epi_ale_analysis/try/boot/{filename}', index_col=None).values
    for i in bad_smiles:
        ecfp_file = ecfp_file[ecfp_file[:, 0] != i]
    print(ecfp_file.shape)
    ecfp_epi_uncertainty = abs(ecfp_file[:, 2])
    ecfp_sort_file = np.vstack((smiles, true_value, predicted_value, ecfp_epi_uncertainty)).T
    ecfp_sort_file = ecfp_sort_file[np.argsort(ecfp_sort_file[:, 3])]

    for j in range(100):
        percent = 100 - j  # 100, 99, 98, 97.... 1
        amount = n_mols * (percent/100)
        y_true = ecfp_sort_file[:int(amount), 1]
        y_pred = ecfp_sort_file[:int(amount), 2]
        percent_mae = mean_absolute_error(y_true, y_pred)
        ecfp_mae.append(percent_mae)

    plt.plot(range(100), ecfp_mae)
    name_.append(filename)

# plt.legend(['original', 'oracle', 'ecfp', 'fcfp'], loc='lower left')
plt.legend(['oracle', 'epi', 'ale', 'total'] + name_, loc='lower left')
plt.grid()
# plt.ylim(0.0, 1.0)
# plt.xlim(0, 100)

plt.ylabel('MAE')
plt.xlabel(f'take away __% of molecule (out of {n_mols})')
plt.title(f'Confidence Curve_{name}')
plt.show()


