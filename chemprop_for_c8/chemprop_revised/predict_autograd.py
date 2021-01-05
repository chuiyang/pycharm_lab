import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from chemprop.data.scaler import StandardScaler
from chemprop.models import build_model
from chemprop.utils import load_args, load_checkpoint, load_scalers
from utils import make_points, gradient
from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions
from torch.autograd import Variable as v
import math


def predict_autograd(test_data):
    checkpoint_path = 'saved_models/qm9_ens_seed60/fold_0/model_0/model.pt'
    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None

    for k in ['encoder.encoder.cached_zero_vector', 'encoder.encoder.W_i.weight', 'encoder.encoder.W_h.weight',
              'encoder.encoder.W_o.weight', 'encoder.encoder.W_o.bias']:
        loaded_state_dict.pop(k, None)

    # Build model
    model = build_model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            print(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            # print(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    model.eval()
    test_data = v(torch.from_numpy(test_data).float(), requires_grad=True)

    # with torch.no_grad():
    model_preds, ale_pred = model(test_data)
    ale_pred = torch.exp(ale_pred)

    model_preds.backward()

    if scaler is not None:
        model_preds = scaler.inverse_transform(model_preds.detach())
        ale_pred = scaler.inverse_transform_variance(ale_pred.detach())



    model_preds = np.array(model_preds.tolist(), dtype=np.float)
    ale_pred = np.array(ale_pred.tolist(), dtype=np.float)
    grad_rms = torch.sqrt(torch.sum(torch.square(test_data.grad.data))/1000).numpy()
    grad_max = torch.max(torch.sqrt(torch.square(test_data.grad.data))).numpy()


    return model_preds, ale_pred, grad_rms, grad_max


if __name__ == '__main__':
    # args = parse_predict_args()
    file_path = f'./saved_models/pred_output/seed60_test/hidden_vector_0.npy'
    mol_fp = np.load(file_path)
    # mol_i = mol_fp[0, :]
    print(mol_fp.shape)
    grad_rmss = []
    grad_maxx = []
    # smiles = pd.read_csv('./saved_models/qm9_ens_woN/fold_0/qm9_N.csv').values[:, 0]
    smiles = pd.read_csv('./saved_models/pred_output/seed60_test/test_full.csv').values[:, 0]
    true = pd.read_csv('./saved_models/pred_output/seed60_test/test_full.csv').values[:, 1]
    for (i, smile, t) in zip(range(mol_fp.shape[0]), smiles, true):  # mol_fp.shape[0]

        mol_i = mol_fp[i, :]
        pred, ale, grad_rms, grad_max = predict_autograd(mol_i)
        grad_rmss.append(grad_rms)
        grad_maxx.append(grad_max)
        print(smile, i+2, grad_rms, grad_max, pred, ale)  # , pred[0], ale[0]

    pd.DataFrame(np.array([list(smiles), grad_rmss, grad_maxx], dtype=np.object).T, index=None, columns=['smiles', 'grad_rms', 'grad_max']).\
        to_csv('./saved_models/pred_output/morgan_test/grad_autograd.csv', index=False)

