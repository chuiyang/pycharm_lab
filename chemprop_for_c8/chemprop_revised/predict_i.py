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


def predict_i(test_data):
    checkpoint_path = 'saved_models/qm9_ens_seed60/fold_0/model_0/model.pt'
    # state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


    # features_scaler = StandardScaler(state['features_scaler']['means'],
    #                                  state['features_scaler']['stds'],
    #                                  replace_nan_token=0) if state['features_scaler'] is not None else None    # Load model and args
    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None

    for k in ['encoder.encoder.cached_zero_vector', 'encoder.encoder.W_i.weight', 'encoder.encoder.W_h.weight',
              'encoder.encoder.W_o.weight', 'encoder.encoder.W_o.bias']:
        loaded_state_dict.pop(k, None)

    # if current_args is not None:
    #     args = current_args
    #
    # args.cuda = cuda if cuda is not None else args.cuda

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
    # model.load_state_dict(pretrained_state_dict)

    # index = torch.from_numpy(np.arange(0, len(test_data))).float()
    model.eval()
    test_data = torch.from_numpy(test_data).float()

    with torch.no_grad():
        model_preds, ale_pred = model(test_data)
        ale_pred = torch.exp(ale_pred)


    if scaler is not None:
        model_preds = scaler.inverse_transform(model_preds.detach())
        ale_pred = scaler.inverse_transform_variance(ale_pred.detach())

    model_preds = np.array(model_preds.tolist(), dtype=np.float)
    ale_pred = np.array(ale_pred.tolist(), dtype=np.float)
    # model_preds = model_preds.data.numpy()
    # ale_pred = ale_pred.data.numpy()
    return model_preds, ale_pred


if __name__ == '__main__':
    # args = parse_predict_args()
    file_path = f'./saved_models/pred_output/seed60_test/hidden_vector_0.npy'
    mol_fp = np.load(file_path)
    # mol_i = mol_fp[0, :]
    print(mol_fp.shape)
    grad_rmss = []
    grad_maxx = []
    # smiles = pd.read_csv('./saved_models/qm9_ens_woN/fold_0/qm9_N.csv').values[:, 0]
    smiles = pd.read_csv('./saved_models/pred_output/seed60_test/test_pred.csv').values[:, 0]
    for i, smile in zip(range(mol_fp.shape[0]), smiles):  # mol_fp.shape[0]
        mol_i = mol_fp[i, :]
        # print(mol_i[:200])
        points = make_points(mol_i)
        # print(points[0])
        pred, ale = predict_i(points)
        grad_rms, grad_max = gradient(pred)
        grad_rmss.append(grad_rms)
        grad_maxx.append(grad_max)
        print(smile, i+2, grad_rms, grad_max, pred[0, 0], ale[0, 0])  # , pred[0], ale[0]
    pd.DataFrame(np.array([list(smiles), grad_rmss, grad_maxx], dtype=np.float).T, index=None, columns=['smiles', 'grad_rms', 'grad_max']).\
        to_csv('./saved_models/pred_output/woN_test/grad_m.csv', index=False)

