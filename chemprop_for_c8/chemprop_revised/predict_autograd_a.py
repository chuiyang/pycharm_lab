import numpy as np
import pandas as pd
import torch
from chemprop.data.scaler import StandardScaler
from chemprop.models.model_autograd_a import build_model_autograd_a
from chemprop.features.featurization_autograd_a import MolGraph_autograd_a, mol2graph_autograd_a
import csv
from torch.autograd import Variable as v


def predict_autograd_a(smile, checkpoint_path):
    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None

    mol_smile = mol2graph_autograd_a(smile, args)
    b_num = mol_smile.f_bonds.shape[0]-1  # vector number = total - cached_zero_vector(1)
    if b_num == 0:
        return np.zeros((1, 1)), 0, 0, 0
    # print('f_bond', mol_smile.f_bonds)

    # Build model
    model = build_model_autograd_a(args)
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

    # load bond_vec
    mol_graph = mol2graph_autograd_a(smile, args)
    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, smiles_batch = mol_graph.get_components()

    f_bonds = v(f_bonds, requires_grad=True)
    model_preds, ale_pred = model(smile, f_bonds)
    ale_pred = torch.exp(ale_pred)
    model_preds.backward()

    atom_grad = []
    for i in torch.square(f_bonds.grad.data):
        atom_grad.append((torch.sum(i).numpy()/len(i))**(1/2))
    atom_max = max(atom_grad)
    atom_rms = sum(atom_grad[1:])/len(atom_grad[1:])

    if scaler is not None:
        model_preds = scaler.inverse_transform(model_preds.detach())
        ale_pred = scaler.inverse_transform_variance(ale_pred.detach())

    model_preds = np.array(model_preds.tolist(), dtype=np.float)
    ale_pred = np.array(ale_pred.tolist(), dtype=np.float)
    return model_preds, ale_pred, atom_rms, atom_max


if __name__ == '__main__':
    file_path = './saved_models/qm9_ens_woN/fold_0/qm9_N.csv'
    smiles = pd.read_csv(file_path).values[:, 0]
    for i in range(0, 2):
        checkpoint_path = f'saved_models/qm9_ens_woN/fold_0/model_{i}/model.pt'
        test_pred = pd.read_csv('./saved_models/pred_output/woN_test/test_pred.csv').values[:, :]
        pred = test_pred[:, 1]
        ale = test_pred[:, 2]
        epi = test_pred[:, 3]
        true = pd.read_csv('./saved_models/pred_output/woN_test/test_full.csv').values[:, 1]
        error = abs(pred - true)

        with open(f'./saved_models/pred_output/woN_test/autograd_a_{i}.csv', 'w+', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['smiles', 'rms_grad', 'max_grad', 'i_error'])
            for (smile, t) in zip(smiles, true):
                pred, ale, atom_rms, atom_max = predict_autograd_a([smile], checkpoint_path)
                print(smile, pred, ale, atom_rms, atom_max, abs(float(pred) - t))
                writer.writerow([smile, atom_rms, atom_max, abs(float(pred) - t)])
