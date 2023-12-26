#!/usr/bin/python
import time
from dataclasses import asdict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from uncond_sggen.model import SceneGraphGen
from uncond_sggen.utils import save_model
from uncond_sggen.parse import AttributeDict


def compute_NLL_dataset(dataset, hyperparam_str):

    for field in asdict(model):
        getattr(model, field).eval()
    graph_dataloader = create_dataloader_only_train(params, dataset, ordering)
    train_dataset = Graph_sequence_sampler(dataset, params, ordering)
    X_all = train_dataset.X_all
    F_all = train_dataset.F_all
    len_all = train_dataset.__len__()

    node_softmax = nn.Softmax(dim=1)
    edge_softmax = nn.Softmax(dim=2)
    idx_range = torch.Tensor(np.arange(params.max_num_node)).to(DEVICE).long()
    nll_list = []
    for X, F, idx in zip(X_all, F_all, np.arange(len_all)):
        prior_score = prior[X[0] - 1]
        first_node_nll = -math.log(prior_score)
        print(idx)
        # INPUTS
        data = train_dataset.__getitem__(idx)
        # GRU_graph
        Xin_ggru = torch.Tensor(data['Xin_ggru']).to(DEVICE).long().unsqueeze(0)
        Fto_in_ggru = torch.Tensor(data['Fto_in_ggru']).to(DEVICE).long().unsqueeze(0)
        Ffrom_in_ggru = torch.Tensor(data['Ffrom_in_ggru']).to(DEVICE).long().unsqueeze(0)
        # GRU_edge
        Xin_egru1 = torch.Tensor(data['Xin_egru1']).to(DEVICE).long().unsqueeze(0)
        Xin_egru2 = torch.Tensor(data['Xin_egru2']).to(DEVICE).long().unsqueeze(0)
        Fto_in_egru = torch.Tensor(data['Fto_in_egru']).to(DEVICE).long().unsqueeze(0)
        Ffrom_in_egru = torch.Tensor(data['Ffrom_in_egru']).to(DEVICE).long().unsqueeze(0)
        Fto_in_egru_shifted = torch.Tensor(data['Fto_in_egru_shifted']).to(DEVICE).long().unsqueeze(0)
        # OUTPUTS
        # MLP_node
        Xout_mlp = torch.Tensor(data['Xout_mlp']).to(DEVICE).long().squeeze(0)
        # GRU_edge
        Fto_out_egru = torch.Tensor(data['Fto_out_egru']).to(DEVICE).long().unsqueeze(0)
        Ffrom_out_egru = torch.Tensor(data['Ffrom_out_egru']).to(DEVICE).long().unsqueeze(0)
        num_edges = data['num_edges']
        seq_len = torch.nonzero(Xout_mlp + 1)[-1] + 1

        # -------------------RUN GRU_graph-----------------------
        # input = concatenated X, F_to, F_from
        Xin_ggru = node_emb(Xin_ggru)
        Fto_in_ggru = edge_emb(Fto_in_ggru)
        Fto_in_ggru = Fto_in_ggru.contiguous().view(Fto_in_ggru.shape[0], Fto_in_ggru.shape[1], -1)
        Ffrom_in_ggru = edge_emb(Ffrom_in_ggru)
        Ffrom_in_ggru = Ffrom_in_ggru.contiguous().view(Ffrom_in_ggru.shape[0], Ffrom_in_ggru.shape[1], -1)
        gru_graph_input = torch.cat((Xin_ggru, Fto_in_ggru, Ffrom_in_ggru), 2)
        # initial hidden state gru_graph
        gru_graph1.hidden = gru_graph1.init_hidden(batch_size=1)
        gru_graph2.hidden = gru_graph2.init_hidden(batch_size=1)
        gru_graph3.hidden = gru_graph3.init_hidden(batch_size=1)
        # scripts the GRU_graph
        hg1 = gru_graph1(gru_graph_input, input_len=seq_len)
        hg2 = gru_graph2(gru_graph_input, input_len=seq_len)
        hg3 = gru_graph3(Xin_ggru, input_len=seq_len)

        # ----------------RUN MLP_node---------------------------
        X_pred = node_softmax(mlp_node(hg3).squeeze())[:seq_len, :]
        Xout_mlp = Xout_mlp[:seq_len]
        node_scores = X_pred[idx_range[:seq_len], Xout_mlp]
        node_nll = compute_NLL(node_scores)
        # ---------------RUN GRU_edge----------------------------
        # Last node produces EOS. for last step, GRU_edge is not scripts
        edge_seq_len = seq_len - 1
        Xin_egru1 = node_emb(Xin_egru1)
        Xin_egru2 = node_emb(Xin_egru2)
        Fto_in_egru = edge_emb(Fto_in_egru)
        Ffrom_in_egru = edge_emb(Ffrom_in_egru)
        Fto_in_egru_shifted = edge_emb(Fto_in_egru_shifted)
        if edge_supervision:
            gru_edge_input1 = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru, Fto_in_egru_shifted), 3)
            gru_edge_input2 = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru), 3)
        else:
            gru_edge_input1 = torch.cat((Fto_in_egru, Ffrom_in_egru), 3)
            gru_edge_input2 = torch.cat((Fto_in_egru, Ffrom_in_egru), 3)

        gru_edge_input1 = pack_padded_sequence(gru_edge_input1, edge_seq_len, batch_first=True,
                                               enforce_sorted=False).data
        gru_edge_input2 = pack_padded_sequence(gru_edge_input2, edge_seq_len, batch_first=True,
                                               enforce_sorted=False).data
        edge_batch_size = gru_edge_input1.shape[0]
        # initial hidden state for gru_edge
        gru_edge_hidden1 = hg1[:, 0:params.max_num_node - 1, :]
        gru_edge_hidden2 = hg2[:, 0:params.max_num_node - 1, :]
        # merge 2nd dimension into batch dimension by packing
        gru_edge_hidden1 = pack_padded_sequence(gru_edge_hidden1, edge_seq_len, batch_first=True,
                                                enforce_sorted=False).data
        gru_edge_hidden2 = pack_padded_sequence(gru_edge_hidden2, edge_seq_len, batch_first=True,
                                                enforce_sorted=False).data
        gru_edge_hidden1 = torch.unsqueeze(gru_edge_hidden1, 0)
        gru_edge_hidden2 = torch.unsqueeze(gru_edge_hidden2, 0)
        if params.egru_num_layers > 1:
            gru_edge_hidden1 = torch.cat((gru_edge_hidden1, torch.zeros(params.egru_num_layers - 1, edge_batch_size,
                                                                        gru_edge_hidden1.shape[2]).to(DEVICE)), 0)
            gru_edge_hidden2 = torch.cat((gru_edge_hidden2, torch.zeros(params.egru_num_layers - 1, edge_batch_size,
                                                                        gru_edge_hidden2.shape[2]).to(DEVICE)), 0)
        gru_edge1.hidden = gru_edge_hidden1
        gru_edge2.hidden = gru_edge_hidden2

        # scripts gru_edge
        Ffrom_pred = edge_softmax(gru_edge1(gru_edge_input1)[:, :edge_seq_len, :])
        Fto_pred = edge_softmax(gru_edge2(gru_edge_input2)[:, :edge_seq_len, :])
        Fto_out_egru = pack_padded_sequence(Fto_out_egru, edge_seq_len, batch_first=True,
                                            enforce_sorted=False).data[:, :edge_seq_len]
        Ffrom_out_egru = pack_padded_sequence(Ffrom_out_egru, edge_seq_len, batch_first=True,
                                              enforce_sorted=False).data[:, :edge_seq_len]
        edge_nll = 0
        for i in range(edge_seq_len):
            edge_score = Fto_pred[i][idx_range[:edge_seq_len], Fto_out_egru[i]][:i + 1]
            edge_nll += compute_NLL(edge_score)
            edge_score = Ffrom_pred[i][idx_range[:edge_seq_len], Ffrom_out_egru[i]][:i + 1]
            edge_nll += compute_NLL(edge_score)

        nll = (first_node_nll + node_nll + edge_nll) / X.shape[0]
        nll_list.append(nll.cpu().detach().numpy())
