#!/usr/bin/python
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from uncond_sggen.parser import AttributeDict


class GraphGRU(nn.Module):
    """
    The rnn which computes graph-level hidden state at each step i,
    from X(i-1), F(i-1) and hidden state from edge level rnn.
    1 layer at input, rnn with num_layers, output hidden state
    """

    def __init__(self, model_args: AttributeDict, is_3=False):
        super(GraphGRU, self).__init__()
        self.model_args = model_args
        # Define the architecture
        self.max_num_node = model_args.max_num_node
        # self.input_size = 448 if not is_3 else model_args.node_emb_size
        if not is_3:
            self.input_size = (self.max_num_node-1)*model_args.edge_emb_size*2 + model_args.node_emb_size
        else:
            self.input_size = model_args.node_emb_size

        self.hidden_size = model_args.ggru_hidden_size
        self.embedding_size = model_args.ggru_emb_size
        self.bias_constant = model_args.bias_constant
        self.num_layers = model_args.ggru_num_layers
        # input
        # print('model 1', self.input_size, self.embedding_size)
        self.input = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU()
        )
        # rnn
        self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True)
        # initialization
        self.hidden = None
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, self.bias_constant)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    # Initial state of GRU_graph is 0.
    def init_hidden(self, batch_size):
        hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.model_args.device)
        return hidden_init

    def forward(self, input_raw, input_len, pack=False):
        # input
        input_ggru = self.input(input_raw)
        if pack:
            input_packed = pack_padded_sequence(input_ggru, input_len, batch_first=True, enforce_sorted=False)
        else:
            input_packed = input_ggru
        # rnn
        output, self.hidden = self.rnn(input_packed, self.hidden)
        if pack:
            output, seq_len = pad_packed_sequence(output, batch_first=True, padding_value=0.0,
                                                  total_length=self.max_num_node)
        return output


class NodeMLP(nn.Module):
    """
    2 layer Multilayer perceptron with sigmoid output to get node categories.
    2 layered fully connected with ReLU
    """

    def __init__(self, model_args: AttributeDict):
        super(NodeMLP, self).__init__()
        self.h_graph_size = model_args.mlp_input_size
        self.embedding_size = model_args.mlp_emb_size
        self.node_size = model_args.mlp_out_size
        self.output = nn.Sequential(
            nn.Linear(self.h_graph_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.node_size)
        )
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, h):
        y = self.output(h)
        return y


class EdgeGRU(nn.Module):
    """
    Sequential NN which outputs the edge categories F(i) using GRU_graph hidden state and X(i)
    1 layer at input, rnn with hidden layers, 2 layer at output
    """

    def __init__(self, model_args: AttributeDict, is_2 = False):
        super(EdgeGRU, self).__init__()

        # Define the architecture
        self.input_size = model_args.egru_input_size1
        if is_2: self.input_size-=8
        self.embedding_size = model_args.egru_emb_input_size
        self.hidden_size = model_args.egru_hidden_size
        self.num_layers = model_args.egru_num_layers
        self.emb_edge_size = model_args.egru_emb_output_size
        self.edge_size = model_args.egru_output_size
        self.bias_constant = model_args.bias_constant
        self.h_edge_size = model_args.egru_emb_input_size
        ## self.h_edge_size = 64

        # input
        # print('model inp emb', self.input_size, self.embedding_size)
        self.input = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU()
        )
        # gru
        self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.h_edge_size,
                          num_layers=self.num_layers, batch_first=True)
        # outputs from the gru
        self.output = nn.Sequential(
            nn.Linear(self.h_edge_size, self.emb_edge_size),
            nn.ReLU(),
            nn.Linear(self.emb_edge_size, self.edge_size)
        )
        # initialization
        self.hidden = None
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, self.bias_constant)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input_raw):
        # input
        # print('model inp raw', input_raw.size(), self.input_size, self.embedding_size)
        input_egru = self.input(input_raw)
        # rnn
        output_raw, self.hidden = self.rnn(input_egru, self.hidden)
        # output
        output = self.output(output_raw)
        return output


@dataclass
class SceneGraphGen:
    node_embedding: nn.Embedding
    edge_embedding: nn.Embedding
    gru_graph1: GraphGRU
    gru_graph2: GraphGRU
    gru_graph3: GraphGRU
    mlp_node: NodeMLP
    gru_edge1: EdgeGRU
    gru_edge2: EdgeGRU
