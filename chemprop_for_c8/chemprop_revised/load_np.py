import numpy as np
import matplotlib.pyplot as plt

# morgan 15個model的hidden都是一樣的
file_path = f'./saved_models/pred_output/qm9_ens_morgan_test/hidden_vector_0.npy'
# file_path = f'./saved_models/pred_output/seed60_test/hidden_vector_0.npy'

num = np.load(file_path)

print(num.shape)

for i, k in enumerate(num):
    print(k[:100], np.mean(k), np.var(k)) if i == 0 else None
    # if i == 0:
    #     plt.bar(range(len(k)), k, 1)
    #     plt.xlim([0, 1000])
    #     plt.ylim([0, 1])
    #     plt.show()

# file_path = f'./saved_models/pred_output/seed60_test_nfs/hidden_vector_0.npy'
# # file_path = f'./saved_models/pred_output/test/hidden_vector_0.npy'
#
# num = np.load(file_path)
#
# print(num.shape)
#
# for i, k in enumerate(num):
#     print(k[:100], np.mean(k), np.var(k)) if i == 0 else None
#     if i == 0:
#         plt.bar(range(len(k)), k, 1)
#         plt.show()


