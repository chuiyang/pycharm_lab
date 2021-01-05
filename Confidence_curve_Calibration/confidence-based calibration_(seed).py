import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import os
# original
# pred_hf = '../../gpu_output/qm9_ens_seed40/qm9_ens_test.csv'   # gabriele pred hf & unc
# molecules = pd.read_csv(pred_hf, index_col=None).values
molecules = pd.read_csv('../prove_ale_epi_(knn)/knn_GCNN/knn_zinc_test/test_pred.csv', index_col=None).values

mean = molecules[:, 1]
var_ale = molecules[:, 2]
var_epi = molecules[:, 3]
var_total = molecules[:, 2] + molecules[:, 3]

# true_hf = '../../gpu_output/qm9_ens_seed40/test_full.csv'    # true hf file
# true_file = pd.read_csv(true_hf, index_col=None).values[:, :]
true_file = pd.read_csv('../prove_ale_epi_(knn)/knn_GCNN/knn_zinc_test/test_full.csv', index_col=None).values
true = true_file[:, 1]

for var in var_epi, var_ale, var_total:
    print(var.shape, len(mean))
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
        cali.append(count/len(mean))
    plt.plot(k_bin, cali, linewidth=2)


# # linear model predicted unc
# path = '../../cpu_output/try/epi_output_seed40_a.csv'
# name = path.split('/')[-1]
#
# pred_unc = pd.read_csv(path, index_col=None).values[:, 2]
# print(pred_unc.shape)
# cali = []
# num = []
# k_bin = np.linspace(0, 1, 20, endpoint=False)
# for j, con in enumerate(k_bin):
#     count = 0
#     for m, v, t in zip(mean, pred_unc, true):
#         l_, u_ = scipy.stats.norm.interval(con, loc=m, scale=v**(1/2))
#         if l_ < t < u_:
#             count += 1
#     num.append(count)
#     cali.append(count/13083)
# plt.plot(k_bin, cali, linewidth=2)


# plt.legend(['epi', 'ale', 'total', name], loc='upper left')
plt.legend(['epi', 'ale', 'total'], loc='upper left')
plt.plot([0, 1], [0, 1], '--', c='gray')
# plt.title(f'{name}')
plt.grid()
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 1.0)
plt.xlabel('confidence interval')
plt.ylabel('% of molecules fall in CI')
plt.show()






