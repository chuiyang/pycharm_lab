import numpy as np
import pandas as pd
import torch
from chemprop.data.scaler import StandardScaler
from chemprop.models.model_a import build_model_a
from utils import gradient_a
from chemprop.features.featurization_a import MolGraph_a, mol2graph_a
import csv


def predict_a(smile):

    checkpoint_path = 'saved_models/qm9_ens_woN/fold_0/model_0/model.pt'
    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None

    mol_smile = mol2graph_a([smile], args)
    b_num = mol_smile.f_bonds.shape[0]-1  # vector number = total - cached_zero_vector(1)
    if b_num == 0:
        return np.zeros((1, 1)), 0, 0
    # print('f_bond', mol_smile.f_bonds)

    # Build model
    model = build_model_a(args)
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

    with torch.no_grad():
        model_preds, ale_pred = model([smile]*b_num*147*2, None)
        ale_pred = torch.exp(ale_pred)

    if scaler is not None:
        model_preds = scaler.inverse_transform(model_preds)
        ale_pred = scaler.inverse_transform_variance(ale_pred)
    model_preds = np.array(model_preds.tolist(), dtype=np.float)
    ale_pred = np.array(ale_pred.tolist(), dtype=np.float)
    # model_preds = model_preds.data.numpy()
    # ale_pred = ale_pred.data.numpy()
    # print(model_preds)
    # print(ale_pred)
    return model_preds, ale_pred, b_num


if __name__ == '__main__':
    file_path = './saved_models/qm9_ens_woN/fold_0/qm9_N.csv'
    # file_path = './saved_models/pred_output/seed60_outlier/outlier_short.csv'
    smiles = pd.read_csv(file_path).values[:, 0]
    smiles = ['C', 'O']
    with open('./saved_models/pred_output/woN_test/grad_a.csv', 'w+', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['smiles', 'max_grad', 'avg_grad'])
        for smile in smiles:
            pred, ale, b_num = predict_a(smile)
            avg_grad, max_grad = gradient_a(pred, b_num)
            print(smile, avg_grad, max_grad, pred[0, 0])
            writer.writerow([smile, avg_grad, max_grad])
    # grad_rmss.append(grad_rms)
    # print(i+2, grad_rms)  # , pred[0], ale[0]
    # pd.DataFrame(np.array(grad_rmss, dtype=np.float).reshape((-1, 1)), index=None, columns=['model_0']).\
    #     to_csv('./saved_models/pred_output/test/model_0.csv', index=False)

