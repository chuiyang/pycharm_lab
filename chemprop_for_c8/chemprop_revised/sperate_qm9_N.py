import pandas as pd
import numpy as np

# file = pd.read_csv('./saved_models/qm9_ens_woN/fold_0/qm9_N.csv')
# header = list(file.columns)
# file = file.values
# print(header)
# print(file.shape)
#
# sli = round(file.shape[0]/5)
# print(sli)
#
#
# total = []
# for i in range(5):
#     if i == 4:
#         file_i = file[i * sli:]
#     else:
#         file_i = file[i*sli:(i+1)*sli]
#     pd.DataFrame(file_i, columns=header, index=None).to_csv(f'./saved_models/qm9_ens_woN/fold_0/qm9_N_{i}.csv', index=False)
#     print(i*sli, (i+1)*sli, file_i.shape)
#     total.append(int(file_i.shape[0]))
#
# print(sum(total))



def read_csv(i):
    return pd.read_csv(f'./saved_models/pred_output/woN_test/grad_a_{i}.csv').values

header = pd.read_csv(f'./saved_models/pred_output/woN_test/grad_a_0.csv').columns

file = np.vstack([read_csv(i) for i in range(5)])

print(file.shape)

pd.DataFrame(file, columns=header, index=None).to_csv('./saved_models/pred_output/woN_test/grad_a.csv', index=False)
