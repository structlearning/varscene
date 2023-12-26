#!/usr/bin/python
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from uncond_sggen.model import SceneGraphGen
from uncond_sggen.utils import save_model, AttributeDict


class SceneGraphGenTrainer:

    def __init__(
            self,
            train_args: AttributeDict,
            model_args: AttributeDict,
            data_loader: DataLoader,
            model: SceneGraphGen
    ):
        self.train_args = train_args
        self.model_args = model_args
        self.data_loader = data_loader
        self.model = model
        self.device = self.model_args.device

    def __forward_pass_gru_graph(
            self,
            data: Dict[str, torch.Tensor],
    ):
        device = self.device
        Xin_ggru = data['Xin_ggru'].to(device).long()
        Fto_in_ggru = data['Fto_in_ggru'].to(device).long()
        Ffrom_in_ggru = data['Ffrom_in_ggru'].to(device).long()
        seq_len = data['len'].to(device).float()
        # -------------------RUN GRU_graph-----------------------
        # print('x', Xin_ggru.size())
        # input = concatenated X, F_to, F_from
        # print('ft', Fto_in_ggru.size())
        # print('ff', Ffrom_in_ggru.size())
        Xin_ggru = self.model.node_embedding(Xin_ggru)
        Fto_in_ggru = self.model.edge_embedding(Fto_in_ggru)
        Ffrom_in_ggru = self.model.edge_embedding(Ffrom_in_ggru)
        Fto_in_ggru = Fto_in_ggru.contiguous().view(Fto_in_ggru.shape[0],
                                                    Fto_in_ggru.shape[1], -1)
        Ffrom_in_ggru = Ffrom_in_ggru.contiguous().view(Ffrom_in_ggru.shape[0],
                                                        Ffrom_in_ggru.shape[1], -1)
        gru_graph_input = torch.cat((Xin_ggru, Fto_in_ggru, Ffrom_in_ggru), 2)
        # initial hidden state gru_graph
        # scripts the GRU_graph
        # print('x', Xin_ggru.size())
        # print('ft', Fto_in_ggru.size())
        # print('ff', Ffrom_in_ggru.size())
        #  print('gp', gru_graph_input.size())
        self.model.gru_graph3.hidden = self.model.gru_graph3.init_hidden(batch_size=self.train_args.batch_size)
        # print('trainer 3')
        ## hg3 = self.model.gru_graph3(gru_graph_input, input_len=seq_len)
        hg3 = self.model.gru_graph3(Xin_ggru, input_len=seq_len)
        self.model.gru_graph1.hidden = self.model.gru_graph1.init_hidden(batch_size=self.train_args.batch_size)
        self.model.gru_graph2.hidden = self.model.gru_graph2.init_hidden(batch_size=self.train_args.batch_size)
        # print('trainer 1', gru_graph_input.size(), seq_len.size())
        hg1 = self.model.gru_graph1(gru_graph_input, input_len=seq_len)
        # print('trainer 2')
        hg2 = self.model.gru_graph2(gru_graph_input, input_len=seq_len)

        return hg1, hg2, hg3

    def __forward_pass_node_mlp(
            self,
            hg3: torch.Tensor
    ):

        X_pred = self.model.mlp_node(hg3)
        X_pred = X_pred.permute(0, 2, 1)
        return X_pred

    def __forward_pass_gru_edge(
            self,
            data: Dict[str, torch.Tensor],
            hg1: torch.Tensor,
            hg2: torch.Tensor
    ):
        device = self.device

        Xin_egru1 = data['Xin_egru1'].to(device).long()
        Xin_egru2 = data['Xin_egru2'].to(device).long()
        Fto_in_egru = data['Fto_in_egru'].to(device).long()
        Ffrom_in_egru = data['Ffrom_in_egru'].to(device).long()
        Fto_in_egru_shifted = data['Fto_in_egru_shifted'].to(device).long()
        seq_len = data['len'].to(device).float()
        # ---------------RUN GRU_edge----------------------------
        # Last node produces EOS. for last step, GRU_edge is not scripts
        edge_seq_len = seq_len - 1
        Xin_egru1 = self.model.node_embedding(Xin_egru1)
        Xin_egru2 = self.model.node_embedding(Xin_egru2)
        Fto_in_egru = self.model.edge_embedding(Fto_in_egru)
        Ffrom_in_egru = self.model.edge_embedding(Ffrom_in_egru)
        Fto_in_egru_shifted = self.model.edge_embedding(Fto_in_egru_shifted)

        if not self.model_args.graphrnn_baseline:
            gru_edge_input1 = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru, Fto_in_egru_shifted), 3)
            ## gru_edge_input2 = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru, Fto_in_egru_shifted), 3)
            gru_edge_input2 = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru), 3)
        else:
            gru_edge_input = torch.cat((Fto_in_egru, Ffrom_in_egru), 3)
            gru_edge_input1 = gru_edge_input
            gru_edge_input2 = gru_edge_input
            # print('here')
        # print('1 gru_edge_input1 gru_edge_input2', gru_edge_input1.size(), gru_edge_input2.size())
        # merge 2nd dimension into batch dimension by packing
        gru_edge_input1 = pack_padded_sequence(gru_edge_input1, edge_seq_len.cpu(),
                                               batch_first=True, enforce_sorted=False).data
        gru_edge_input2 = pack_padded_sequence(gru_edge_input2, edge_seq_len.cpu(),
                                               batch_first=True, enforce_sorted=False).data
        # print('2 gru_edge_input1 gru_edge_input2', gru_edge_input1.size(), gru_edge_input2.size())
        edge_batch_size = gru_edge_input1.shape[0]
        # initial hidden state for gru_edge
        gru_edge_hidden1 = hg1[:, 0:self.model_args.max_num_node - 1, :]
        gru_edge_hidden2 = hg2[:, 0:self.model_args.max_num_node - 1, :]
        # print('3 gru_edge_input1 gru_edge_input2', gru_edge_input1.size(), gru_edge_input2.size())
        # merge 2nd dimension into batch dimension by packing
        gru_edge_hidden1 = pack_padded_sequence(gru_edge_hidden1, edge_seq_len.cpu(),
                                                batch_first=True, enforce_sorted=False).data
        gru_edge_hidden2 = pack_padded_sequence(gru_edge_hidden2, edge_seq_len.cpu(),
                                                batch_first=True, enforce_sorted=False).data
        # print('4 gru_edge_input1 gru_edge_input2', gru_edge_input1.size(), gru_edge_input2.size())
        gru_edge_hidden1 = torch.unsqueeze(gru_edge_hidden1, 0)
        gru_edge_hidden2 = torch.unsqueeze(gru_edge_hidden2, 0)
        # print('5 gru_edge_input1 gru_edge_input2', gru_edge_input1.size(), gru_edge_input2.size())
        if self.model_args.egru_num_layers > 1:
            gru_edge_hidden1 = torch.cat((gru_edge_hidden1, torch.zeros(self.model_args.egru_num_layers - 1,
                                                                        edge_batch_size,
                                                                        gru_edge_hidden1.shape[2]).to(device)), 0)
            gru_edge_hidden2 = torch.cat((gru_edge_hidden2, torch.zeros(self.model_args.egru_num_layers - 1,
                                                                        edge_batch_size,
                                                                        gru_edge_hidden2.shape[2]).to(device)), 0)
        # print('6 gru_edge_input1 gru_edge_input2', gru_edge_input1.size(), gru_edge_input2.size())
        self.model.gru_edge1.hidden = gru_edge_hidden1
        self.model.gru_edge2.hidden = gru_edge_hidden2
        # gru_edge
        Ffrom_pred = self.model.gru_edge1(gru_edge_input1)
        Fto_pred = self.model.gru_edge2(gru_edge_input2)

        Fto_pred = Fto_pred.permute(0, 2, 1)
        Ffrom_pred = Ffrom_pred.permute(0, 2, 1)

        return Fto_pred, Ffrom_pred

    def __compute_node_loss(
            self,
            data: Dict[str, torch.Tensor],
            X_pred: torch.Tensor,
            ce_loss_node: nn.CrossEntropyLoss
    ):
        seq_len = data['len'].to(self.device).float()
        Xout_mlp = data['Xout_mlp'].to(self.device).long()
        node_loss = ce_loss_node(X_pred, Xout_mlp)
        node_loss = torch.sum(node_loss) / torch.sum(seq_len)

        return node_loss

    def __compute_edge_loss(
            self,
            data: Dict[str, torch.Tensor],
            Fto_pred: torch.Tensor,
            Ffrom_pred: torch.Tensor,
            ce_loss_edge: nn.CrossEntropyLoss
    ):
        seq_len = data['len'].to(self.device).float()
        num_edges = data['num_edges'].to(self.device).float()
        edge_seq_len = seq_len - 1

        Fto_out_egru = data['Fto_out_egru'].to(self.device).long()
        Fto_out_egru = pack_padded_sequence(Fto_out_egru, edge_seq_len.cpu(),
                                            batch_first=True, enforce_sorted=False).data

        Ffrom_out_egru = data['Ffrom_out_egru'].to(self.device).long()
        Ffrom_out_egru = pack_padded_sequence(Ffrom_out_egru, edge_seq_len.cpu(),
                                              batch_first=True, enforce_sorted=False).data

        Fto_edge_loss = ce_loss_edge(Fto_pred, Fto_out_egru)
        Ffrom_edge_loss = ce_loss_edge(Ffrom_pred, Ffrom_out_egru)
        edge_loss = Fto_edge_loss + Ffrom_edge_loss
        edge_loss = torch.sum(edge_loss) / torch.sum(num_edges)

        return edge_loss

    def train(self):
        model = self.model
        train_args = self.train_args
        model_args = self.model_args
        # initialize optimizer
        node_parameters = list(model.mlp_node.parameters()) + list(model.gru_graph3.parameters())
        optimizer_node = optim.Adam(node_parameters, lr=train_args.node_lr_init)
        scheduler_node = StepLR(optimizer_node, step_size=train_args.node_step_decay_epochs, gamma=train_args.node_lr_decay)

        edge_parameters = list(model.gru_graph1.parameters()) + list(model.gru_edge1.parameters()) \
                          + list(model.gru_graph2.parameters()) + list(model.gru_edge2.parameters())
        optimizer_edge = optim.Adam(edge_parameters, lr=train_args.edge_lr_init)
        scheduler_edge = StepLR(optimizer_edge, step_size=train_args.edge_step_decay_epochs, gamma=train_args.edge_lr_decay)

        # the outputs are padded with -1. Loss function doesnt compute loss corresponding to index -1
        ce_loss_node = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        ce_loss_edge = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        best_loss = np.inf
        for epoch in range(train_args.epochs):
            time_start = time.time()
            for field in model.__dataclass_fields__:
                getattr(model, field).train()

            train_loss = 0
            for batch_idx, batch in enumerate(self.data_loader):
                # tg = time.time()
                optimizer_node.zero_grad()
                optimizer_edge.zero_grad()

                # forward pass
                hg1, hg2, hg3 = self.__forward_pass_gru_graph(batch)
                X_pred = self.__forward_pass_node_mlp(hg3)
                Fto_pred, Ffrom_pred = self.__forward_pass_gru_edge(batch, hg1, hg2)
                node_loss = self.__compute_node_loss(batch, X_pred, ce_loss_node)
                edge_loss = self.__compute_edge_loss(batch, Fto_pred, Ffrom_pred, ce_loss_edge)

                # backward pass
                node_loss.backward()
                optimizer_node.step()
                scheduler_node.step()
                edge_loss.backward()
                optimizer_edge.step()
                scheduler_edge.step()

                loss = node_loss + edge_loss
                train_loss += loss.cpu().data.numpy()
                # print('%.4fs per batch' % (time.time()-tg))

                if batch_idx % 10 == 0:
                    print('Loss %.4f, node loss %.4f, edge loss %.4f' % (loss.item(), node_loss.item(), edge_loss.item()))

            train_loss = train_loss / (len(self.data_loader.dataset))
            print('train loss, best loss', train_loss, best_loss)
            if train_loss < best_loss:
                best_loss = train_loss
                print('Saving')
                save_model(model=model, model_args=model_args, train_args=train_args)

            time_end = time.time()
            lr_node = optimizer_node.param_groups[0]['lr']
            lr_edge = optimizer_edge.param_groups[0]['lr']
            print('Epoch: ', epoch, 'Training Loss: ', train_loss, 'node lr: ', lr_node, 'edge lr: ', lr_edge, 'time: ',
                  time_end - time_start)
