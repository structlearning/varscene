#!/usr/bin/python
from typing import Tuple
from dataclasses import asdict

import torch
import torch.nn as nn
import numpy as np
from uncond_sggen.parse import AttributeDict
from uncond_sggen.model import SceneGraphGen


def complete_graph(
        eval_args: AttributeDict,
        model_args: AttributeDict,
        model: SceneGraphGen,
        partial_graph: Tuple[np.array, np.array]
):

    X_, F_ = partial_graph
    n_ = X_.shape[0]
    F_ = np.where(F_ == 0, model_args.no_edge_token, F_)
    Fto_ = np.triu(F_, +1)
    Ffrom_ = np.transpose(np.tril(F_, -1))

    num_graphs = 1
    # set models to eval mode
    for field in asdict(model):
        getattr(model, field).eval()

    # intantiate generated graphs
    X = torch.zeros(num_graphs,
                    model_args.max_num_node - 1).to(eval_args.device).long()
    X[:, :n_] = torch.Tensor(X_)
    Fto = torch.zeros(num_graphs,
                      model_args.max_num_node - 1,
                      model_args.max_num_node).to(eval_args.device).long()
    Fto[:, :n_, :n_] = torch.Tensor(Fto_)
    Ffrom = torch.zeros(num_graphs,
                        eval_args.device.max_num_node - 1,
                        eval_args.device.max_num_node).to(eval_args.device).long()
    Ffrom[:, :n_, :n_] = torch.Tensor(Ffrom_)

    # initial edge vector
    init_edges = torch.zeros(num_graphs, model_args.max_num_node - 1).to(eval_args.device).long()
    edge_SOS_token = torch.Tensor([model_args.edge_SOS_token]).to(eval_args.device).long()
    # init gru_graph hidden state
    model.gru_graph1.hidden = model.gru_graph1.init_hidden(num_graphs)
    model.gru_graph2.hidden = model.gru_graph2.init_hidden(num_graphs)
    model.gru_graph3.hidden = model.gru_graph3.init_hidden(num_graphs)

    scene_graph = None
    softmax = nn.Softmax(dim=0)
    for i in range(model_args.max_num_node - 1):
        # update graph info with generated nodes/edges

        Fto_vec = Fto[:, :, i]
        Ffrom_vec = Ffrom[:, :, i]

        Xsample1 = torch.unsqueeze(X[:, i], 1)
        Fto_vec = torch.unsqueeze(Fto_vec, 1)
        Ffrom_vec = torch.unsqueeze(Ffrom_vec, 1)
        Xsample1 = model.node_embedding(Xsample1)
        Fto_vec = model.edge_embedding(Fto_vec)
        Fto_vec = Fto_vec.contiguous().view(Fto_vec.shape[0],
                                            Fto_vec.shape[1], -1)
        Ffrom_vec = model.edge_embedding(Ffrom_vec)
        Ffrom_vec = Ffrom_vec.contiguous().view(Ffrom_vec.shape[0],
                                                Ffrom_vec.shape[1], -1)
        gru_graph_in = torch.cat((Xsample1.float(), Fto_vec.float(), Ffrom_vec.float()), 2)

        # scripts one step of gru_graph
        gru_edge_hidden1 = model.gru_graph1(gru_graph_in, list(np.ones(num_graphs))).data
        gru_edge_hidden2 = model.gru_graph2(gru_graph_in, list(np.ones(num_graphs))).data
        mlp_input = model.gru_graph3(Xsample1.float(), list(np.ones(num_graphs))).data

        # scripts mlp_node and sample next object
        Xscores = model.mlp_node(mlp_input)
        Xscores = torch.squeeze(Xscores)
        if i >= n_ - 1:
            Xsample = torch.multinomial(softmax(Xscores), 1)
            X[:, i + 1] = Xsample + 1
            # exit if EOS token is encountered
            if Xsample.data.cpu().numpy() == model_args.node_EOS_token or i == model_args.max_num_node - 2:
                X = X[:, 0:i]
                Fto = Fto[:, 0:i, 0:i]
                Ffrom = Ffrom[:, 0:i, 0:i]
                X_gen = torch.squeeze(X).cpu().numpy()
                Fto_gen = torch.squeeze(Fto).cpu().numpy()
                Ffrom_gen = torch.squeeze(Ffrom).cpu().numpy()
                Fto_gen = np.where(Fto_gen == model_args.no_edge_token, 0, Fto_gen)
                Ffrom_gen = np.where(Ffrom_gen == model_args.no_edge_token, 0, Ffrom_gen)
                F_gen = Fto_gen + np.transpose(Ffrom_gen)
                scene_graph = X_gen, F_gen
                break

        # get initial hidden state of gru_edge
        if model_args.egru_num_layers > 1:
            gru_edge_hidden1 = torch.cat((gru_edge_hidden1,
                                          torch.zeros(model_args.egru_num_layers - 1,
                                                      gru_edge_hidden1.shape[1],
                                                      gru_edge_hidden1.shape[2]).to(eval_args.device)), 0)
            gru_edge_hidden2 = torch.cat((gru_edge_hidden2,
                                          torch.zeros(model_args.egru_num_layers - 1,
                                                      gru_edge_hidden2.shape[1],
                                                      gru_edge_hidden2.shape[2]).to(eval_args.device)), 0)
        model.gru_edge1.hidden = gru_edge_hidden1
        model.gru_edge2.hidden = gru_edge_hidden2

        # init edge vectors
        Fto_vec = init_edges.clone()
        Ffrom_vec = init_edges.clone()
        for j in range(i + 1):
            # input for gru_in
            x1 = X[:, j]
            x2 = X[:, i + 1]
            fto = Fto_vec[:, j - 1] if j > 0 else edge_SOS_token
            ffrom = Ffrom_vec[:, j - 1] if j > 0 else edge_SOS_token
            # print('Inputs to egru', x1, x2, fto, ffrom)
            x1 = model.node_embedding(x1.view(x1.shape[0], 1))
            x2 = model.node_embedding(x2.view(x2.shape[0], 1))
            fto = model.edge_embedding(fto.view(fto.shape[0], 1))
            ffrom = model.edge_embedding(ffrom.view(ffrom.shape[0], 1))

            gru_edge_in = torch.cat((x1, x2, fto, ffrom), 2)
            # scripts gru_edge and sample next edge
            gru_edge_in1 = torch.cat((x1, x2, fto, ffrom), 2)
            Fto_scores = model.gru_edge2(gru_edge_in1)
            Fto_scores = torch.squeeze(Fto_scores)
            if i >= n_ - 1:
                Fto_sample = torch.multinomial(softmax(Fto_scores), 1)
                Fto_vec[:, j] = torch.squeeze(Fto_sample) + 1
            fto_out = Fto_vec[:, j]
            fto_out = model.edge_emb(fto_out.view(fto_out.shape[0], 1))

            gru_edge_in2 = torch.cat((x1, x2, fto, ffrom, fto_out), 2)
            Ffrom_scores = model.gru_edge1(gru_edge_in2)
            Ffrom_scores = torch.squeeze(Ffrom_scores)
            if i >= n_ - 1:
                Ffrom_sample = torch.multinomial(softmax(Ffrom_scores), 1)
                Ffrom_vec[:, j] = torch.squeeze(Ffrom_sample) + 1
            # update hidden state of gru_edge
            model.gru_edge1.hidden = model.gru_edge1.hidden.data.to(eval_args.device)
            model.gru_edge2.hidden = model.gru_edge2.hidden.data.to(eval_args.device)

        # update hidden state of gru_graph
        model.gru_graph1.hidden = model.gru_graph1.hidden.data.to(eval_args.device)
        model.gru_graph2.hidden = model.gru_graph2.hidden.data.to(eval_args.device)
        model.gru_graph3.hidden = model.gru_graph3.hidden.data.to(eval_args.device)

        if i >= n_ - 1:
            Fto[:, :, i + 1] = Fto_vec
            Ffrom[:, :, i + 1] = Ffrom_vec

    return scene_graph
