#!/usr/bin/python
import random
from typing import List

import numpy as np
from torch.utils.data import Dataset

from uncond_sggen.vg_data.permutation import bfs_ordered, predefined_ordered, random_ordered, motif_based_ordered
from uncond_sggen.parser import AttributeDict


class SceneGraphSequenceSampler(Dataset):
    def __init__(
            self,
            dataset: List,
            model_args: AttributeDict
    ):
        super(SceneGraphSequenceSampler, self).__init__()

        self.n = model_args.max_num_node
        self.permutation = model_args.permutation

        self.X_all = []
        self.F_all = []
        self.len_all = []
        for graph in dataset:
            X = graph[0]
            F = graph[1]
            if self.n >= X.shape[0] >= model_args.min_num_node:
                self.X_all.append(X)
                self.F_all.append(F)
                self.len_all.append(X.shape[0])

        self.node_EOS_token = model_args.node_EOS_token
        self.edge_SOS_token = model_args.edge_SOS_token
        self.no_edge_token = model_args.no_edge_token

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        """
        Inputs:
        X can be 0 for padding, 1 to 150 for each object category
        F can be 0 for padding, 1 to 50 for each edge(relationship) category, 51 for no_edge, 52 edge_SOS_token
        Outputs:
        X can be -1 for padding, 0 to 149 for each object category, 150 for node_EOS_token
        F can be -1 for padding, 0 to 49 for each category, 50 for no_edge
        """
        # get sample
        self.X = self.X_all[idx].copy()
        self.F = self.F_all[idx].copy()
        self.len_item = self.len_all[idx]

        self.__apply_permutation()

        self.F = np.where(self.F == 0, self.no_edge_token, self.F)
        self.Fto = np.triu(self.F, +1)
        self.Ffrom = np.transpose(np.tril(self.F, -1))

        # ------INPUTS OF THE MODEL------------------------
        # GRU_graph
        Xin_ggru, Fto_in_ggru, Ffrom_in_ggru = self.__get_ggru_input()
        # GRU_edge
        Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru, Fto_in_egru_shifted = self.__get_egru_input()

        # ---------OUTPUTS OF THE MODEL------------------------
        # node MLP
        Xout_mlp = self.__get_output_nmlp()
        # GRU edge
        Fto_out_egru, Ffrom_out_egru = self.__get_output_egru()

        return {
            # inputs
            'Xin_ggru': Xin_ggru,
            'Fto_in_ggru': Fto_in_ggru,
            'Ffrom_in_ggru': Ffrom_in_ggru,
            'Xin_egru1': Xin_egru1,
            'Xin_egru2': Xin_egru2,
            'Fto_in_egru': Fto_in_egru,
            'Ffrom_in_egru': Ffrom_in_egru,
            'Fto_in_egru_shifted': Fto_in_egru_shifted,
            # outputs
            'Xout_mlp': Xout_mlp,
            'Fto_out_egru': Fto_out_egru,
            'Ffrom_out_egru': Ffrom_out_egru,
            # sequence lengths
            'len': self.len_item,
            'num_edges': 0.5 * (self.len_item - 1) * self.len_item,
        }

    def __apply_permutation(self):
        # order the graph to a sequence to get permuted X and F
        if self.permutation == 'random':
            self.X, self.F = random_ordered(self.X, self.F)
        elif self.permutation == 'predefined':
            self.X, self.F = predefined_ordered(self.X, self.F)
        elif self.permutation == 'bfs':
            root = random.choice(np.arange(self.X.shape[0]))
            self.X, self.F = bfs_ordered(self.X, self.F, root)
        elif self.permutation == 'motif_based':
            self.X, self.F = motif_based_ordered(self.X, self.F, self.ind_to_classes)
        else:
            raise ValueError('Ordering not understood')

    def __get_ggru_input(self):
        Xin_ggru = np.zeros(self.n)
        Xin_ggru[0:self.len_item] = self.X

        Fto_in_ggru = np.zeros((self.n, self.n))
        Fto_in_ggru[0, 0:self.len_item] = self.edge_SOS_token * np.ones(self.len_item)
        Fto_in_ggru[1:self.len_item, 0:self.len_item] = self.Fto[0:self.len_item - 1, :]
        Fto_in_ggru = np.transpose(Fto_in_ggru)[:, 1:]

        Ffrom_in_ggru = np.zeros((self.n, self.n))
        Ffrom_in_ggru[0, 0:self.len_item] = self.edge_SOS_token * np.ones(self.len_item)
        Ffrom_in_ggru[1:self.len_item, 0:self.len_item] = self.Ffrom[0:self.len_item - 1, :]
        Ffrom_in_ggru = np.transpose(Ffrom_in_ggru)[:, 1:]

        return Xin_ggru, Fto_in_ggru, Ffrom_in_ggru

    def __get_egru_input(self):
        Xseq1 = np.array([self.X[0:self.len_item - 1], ] * (self.len_item - 1)).transpose()
        Xseq1 = np.triu(Xseq1)
        Xin_egru1 = np.zeros((self.n - 1, self.n - 1))
        Xin_egru1[0:self.len_item - 1, 0:self.len_item - 1] = Xseq1
        Xin_egru1[0:self.len_item - 1, 0:self.len_item - 1] = Xseq1
        Xin_egru1 = np.transpose(Xin_egru1)

        Xseq2 = np.array([self.X[1:self.len_item], ] * (self.len_item - 1))
        Xseq2 = np.triu(Xseq2)
        Xin_egru2 = np.zeros((self.n - 1, self.n - 1))
        Xin_egru2[0:self.len_item - 1, 0:self.len_item - 1] = Xseq2
        Xin_egru2 = np.transpose(Xin_egru2)

        Fto_in_egru = np.zeros((self.n - 1, self.n - 1))
        Fto_in_egru[0, 0:self.len_item - 1] = self.edge_SOS_token * np.ones(self.len_item - 1)
        Fto_in_egru[1:self.len_item - 1, 0:self.len_item - 1] \
            = np.triu(self.Fto, +2)[0:self.len_item - 2, 1:self.len_item]
        Fto_in_egru = np.transpose(Fto_in_egru)

        Ffrom_in_egru = np.zeros((self.n - 1, self.n - 1))
        Ffrom_in_egru[0, 0:self.len_item - 1] = self.edge_SOS_token * np.ones(self.len_item - 1)
        Ffrom_in_egru[1:self.len_item - 1, 0:self.len_item - 1] \
            = np.triu(self.Ffrom, +2)[0:self.len_item - 2, 1:self.len_item]
        Ffrom_in_egru = np.transpose(Ffrom_in_egru)

        Fto_in_egru_shifted = np.zeros((self.n - 1, self.n - 1))
        Fto_in_egru_shifted[0:self.len_item - 1, 0:self.len_item - 1] = self.Fto[0:self.len_item - 1, 1:self.len_item]
        Fto_in_egru_shifted = np.transpose(Fto_in_egru_shifted)

        return Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru, Fto_in_egru_shifted

    def __get_output_nmlp(self):
        Xout_mlp = -1 * np.ones(self.n)
        Xout_mlp[0:self.len_item - 1] = self.X[1:] - 1
        Xout_mlp[self.len_item - 1] = self.node_EOS_token

        return Xout_mlp

    def __get_output_egru(self):
        Fto_out_egru = np.zeros((self.n - 1, self.n - 1))
        Fto_out_egru[0:self.len_item - 1, 0:self.len_item - 1] = self.Fto[0:self.len_item - 1, 1:self.len_item]
        Fto_out_egru = np.transpose(Fto_out_egru) - 1

        Ffrom_out_egru = np.zeros((self.n - 1, self.n - 1))
        Ffrom_out_egru[0:self.len_item - 1, 0:self.len_item - 1] = self.Ffrom[0:self.len_item - 1, 1:self.len_item]
        Ffrom_out_egru = np.transpose(Ffrom_out_egru) - 1

        return Fto_out_egru, Ffrom_out_egru
