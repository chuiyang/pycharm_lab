from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout

import pickle

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""
    output_x = []
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim  # 133
        self.bond_fdim = bond_fdim  # 147
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages  # Use messages on atoms instead of messages on bonds, default=False
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.epistemic = args.epistemic
        self.mc_dropout = self.epistemic == 'mc_dropout'
        self.args = args
        self.features_generator = args.features_generator


        if self.features_only or self.features_generator:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Concrete Dropout for Bayesian NN
        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim  # 默認input dim -> bond_fdim 147

        if self.mc_dropout:
            self.W_i = ConcreteDropout(layer=nn.Linear(input_dim, self.hidden_size, bias=self.bias), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd)
        else:
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)  # in 147 out 1000 self.bias-> False (no bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size   # 1000

        # Shared weight matrix across depths (default)
        if self.mc_dropout:
            self.W_h = ConcreteDropout(layer=nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd, depth=self.depth - 1)
            self.W_o = ConcreteDropout(layer=nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd)
        else:
            self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)  # in 1000 out 1000 (no bias)
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)  # in 1000+133 out 1000 (with bias 1000)


    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, smiles_batch = mol_graph.get_components()
        print('f_atoms', f_atoms.shape)
        print(f_atoms[0, :])
        print(f_atoms[1, :])
        print('f_bonds', f_bonds.shape)
        print(f_bonds[0, :])
        print('a2b', a2b)
        print('b2a', b2a)
        print('b2revb', b2revb)
        print('a_scope', a_scope)
        print('b_scope', b_scope)

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        print('smiles_batch', smiles_batch)
        print('f_bonds', f_bonds.shape)
        print('\n\n\n\n')
        # print('input', input.shape)
        message = self.act_func(input)  # num_bonds x hidden_size
        # print('before', message[0, :2])
        # message[0, 0] = message[0, 0] + 0.001
        # print('after', message[0, :2])
        # print('message', message, message.shape)
        # print('\n\n')

        # Message passing
        for depth in range(self.depth - 1):  # GCNN
            # print('depth', depth)
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)  133 + 1000
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
        #        MPNEncoder.output_x.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size  # sum 一個分子中所有原子向量 / 原子數量
                mol_vecs.append(mol_vec)
        #        MPNEncoder.output_x.append(mol_vec)

	#####################################################
        #if len(mol_vecs) < 50:
        #        nnnnvecs = torch.stack(MPNEncoder.output_x, dim=0)
        #        with open('hidden_vector.pkl', 'wb') as f:
        #                pickle.dump(nnnnvecs, f)
        #                print('mol_vecs: ', nnnnvecs)
        #                print('mol_vecs_size: ', nnnnvecs.shape)
	#####################################################
 
        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)   # None, 133
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim  # None, 14 + True(1) * 133  -> 147
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)
        output = self.encoder.forward(batch, features_batch)  # batch is molecular graph
        print('output.shape', output.shape)

        return output
