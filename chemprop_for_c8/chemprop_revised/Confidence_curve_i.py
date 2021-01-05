import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

place = 'woN_test'
pred_hf = f'./saved_models/pred_output/{place}/test_pred.csv'
true_hf = f'./saved_models/pred_output/{place}/test_full.csv'
gradient = f'./saved_models/pred_output/{place}/grad_m_2.csv'

molecules = pd.read_csv(pred_hf, index_col=None).values[:, :]
smiles = molecules[:, 0]
predicted_value = molecules[:, 1]
ale_uncertainty = molecules[:, 2]
epi_uncertainty = molecules[:, 3]


true_file = pd.read_csv(true_hf, index_col=None).values[:, :]
true_value = true_file[:, 1]

max_grad = pd.read_csv(gradient, index_col=None).values[:, 1]
avg_grad = pd.read_csv(gradient, index_col=None).values[:, 0]
n_mols = molecules.shape[0]


# place = 'morgan_test'
# pred_hf = f'./saved_models/pred_output/{place}/every_model_predictions.csv'   # morgan
# true_hf = f'./saved_models/pred_output/{place}/test_full.csv'
# gradient = f'./saved_models/pred_output/{place}/grad_m.csv'
# ale_uncertainty = pd.read_csv(f'./saved_models/pred_output/{place}/every_model_aleatorics.csv',
#                     index_col=None).values[:, 8]  # morgan
# molecules = pd.read_csv(pred_hf, index_col=None).values[:, :]
# smiles = molecules[:, 0]
# predicted_value = molecules[:, 8]
#
# true_file = pd.read_csv(true_hf, index_col=None).values[:, :]
# true_value = true_file[:, 1]
#
# max_grad = pd.read_csv(gradient, index_col=None).values[:, 1]
# avg_grad = pd.read_csv(gradient, index_col=None).values[:, 0]
# n_mols = molecules.shape[0]


true_error = abs(true_value-predicted_value)

# oracle confidence curve
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

# grad(max)
m = np.vstack((smiles, true_value, predicted_value, max_grad)).T
m = m[np.argsort(m[:, 3])]
max_grad_list = []
for j in range(100):
    percent = 100 - j  # 100, 99, 98, 97.... 1
    amount = n_mols * (percent/100)
    y_true = m[:int(amount), 1]
    y_pred = m[:int(amount), 2]
    percent_mae = mean_absolute_error(y_true, y_pred)
    max_grad_list.append(percent_mae)

# grad(avg)
g = np.vstack((smiles, true_value, predicted_value, avg_grad)).T
g = g[np.argsort(g[:, 3])]
avg_grad_list = []
for j in range(100):
    percent = 100 - j  # 100, 99, 98, 97.... 1
    amount = n_mols * (percent/100)
    y_true = g[:int(amount), 1]
    y_pred = g[:int(amount), 2]
    percent_mae = mean_absolute_error(y_true, y_pred)
    avg_grad_list.append(percent_mae)

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

plt.cla()
plt.plot(range(100), ora_mae, c='gray', linestyle='--')
plt.plot(range(100), max_grad_list)
plt.plot(range(100), avg_grad_list)
plt.plot(range(100), aa)
plt.plot(range(100), ee)


plt.legend(['oracle', 'max_grad', 'avg_grad', 'ale', 'epi'], loc='lower left')

plt.grid()
# plt.ylim(0.0, 1.0)
# plt.xlim(0, 100)

plt.ylabel('MAE')
plt.xlabel(f'take away __% of molecule (out of {n_mols})')
plt.title(f'Confidence Curve')
plt.show()


