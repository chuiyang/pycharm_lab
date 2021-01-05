import numpy as np
import pandas as pd

for f in ['test', 'train']:

    test_pred = pd.read_csv(f'./saved_models/pred_output/woN_{f}/{f}_pred.csv').values[:, :]
    pred = test_pred[:, 1]
    ale = test_pred[:, 2]
    epi = test_pred[:, 3]
    true = pd.read_csv(f'./saved_models/pred_output/woN_{f}/{f}_full.csv').values[:, 1]

    def read_csv(i):
        return pd.read_csv(f'./saved_models/pred_output/woN_{f}/autograd_a_{i}.csv').values

    smiles = read_csv(0)[:, 0]
    rms_grad = np.hstack([read_csv(i)[:, 1] for i in range(15)])
    max_grad = np.hstack([read_csv(i)[:, 2] for i in range(15)])

    rms_grad_std = np.std(rms_grad, axis=1).reshape((-1, 1))
    print('rms_grad_std.shape:', rms_grad_std.shape)
    max_grad_std = np.std(max_grad, axis=1).reshape((-1, 1))
    print('max_grad_std.shape:', max_grad_std.shape)

    std = np.hstack([rms_grad_std, max_grad_std, epi, ale, abs(true-pred)])
    pd.DataFrame(std, columns=['rms_grad_std', 'max_grad_std', 'epi', 'ale', 'error'], index=smiles).to_csv(f'./saved_models/pred_output/woN_{f}/autograd_a_std.csv')