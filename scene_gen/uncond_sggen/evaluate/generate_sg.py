#!/usr/bin/python
import os
import pickle
import random
from dataclasses import asdict
from typing import Dict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from uncond_sggen.vg_data.permutation import random_ordered, predefined_ordered, bfs_ordered, motif_based_ordered
from uncond_sggen.model import SceneGraphGen
from uncond_sggen.parser import AttributeDict


def generate_prior_distribution(
        dataset: List,
        model_args: AttributeDict,
        eval_args: AttributeDict,
        # num_iterations: int = 1000
        num_iterations: int = 100
):
    class_dict = dict()
    for it in range(num_iterations):
        for graph in dataset:
            X, F = graph
            # order the graph to a sequence to get permuted X and F
            if model_args.permutation == 'random':
                X, F = random_ordered(X, F)
            elif model_args.permutation == 'predefined':
                X, F = predefined_ordered(X, F)
            elif model_args.permutation == 'bfs':
                root = random.choice(np.arange(X.shape[0]))
                X, F = bfs_ordered(X, F, root)
            elif model_args.permutation == 'motif_based':
                X, F = motif_based_ordered(X, F, eval_args.ind_to_classes)
            else:
                raise ValueError('Ordering not understood')

            first_node = X[0]
            if first_node not in class_dict:
                class_dict[first_node] = 1
            else:
                class_dict[first_node] += 1
        if it%10 == 0:
            print('completed %s iterations for prior construction' % it)

    total = sum(list(class_dict.values()))
    class_dict = {k: v / total for k, v in class_dict.items()}

    os.makedirs(eval_args.eval_path, exist_ok=True)
    with open(os.path.join(eval_args.eval_path, 'prior_distribution.p'), 'wb') as f:
        pickle.dump(class_dict, f)


def generate_one_graph(
        eval_args: AttributeDict,
        model_args: AttributeDict,
        model: SceneGraphGen,
        first_node,
):
    num_graphs = 1
    # set models to eval mode
    ## for field in asdict(model):
    ##     getattr(model, field).eval()

    # intantiate generated graphs
    X = torch.zeros(num_graphs,
                    model_args.max_num_node - 1).to(eval_args.device).long()
    Fto = torch.zeros(num_graphs,
                      model_args.max_num_node - 1,
                      model_args.max_num_node).to(eval_args.device).long()
    Ffrom = torch.zeros(num_graphs,
                        model_args.max_num_node - 1,
                        model_args.max_num_node).to(eval_args.device).long()
    # sample initial object
    Xsample = torch.Tensor([first_node - 1]).long().to(eval_args.device)
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
        X[:, i] = Xsample + 1
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
        ## mlp_input = model.gru_graph3(gru_graph_in, list(np.ones(num_graphs))).data
        mlp_input = model.gru_graph3(Xsample1.float(), list(np.ones(num_graphs))).data

        # scripts mlp_node and sample next object
        Xscores = model.mlp_node(mlp_input)
        Xscores = torch.squeeze(Xscores)
        Xsample = torch.multinomial(softmax(Xscores), 1)
        # exit if EOS token is encountered
        if Xsample.data.cpu().numpy() == model_args.node_EOS_token or i == model_args.max_num_node - 2:
            if i == 0 or i == 1:
                break
            else:
                X = X[:, 0:i]
                Fto = Fto[:, 0:i, 0:i]
                Ffrom = Ffrom[:, 0:i, 0:i]
                X_gen = torch.squeeze(X).cpu().numpy()
                Fto_gen = torch.squeeze(Fto).cpu().numpy()
                Ffrom_gen = torch.squeeze(Ffrom).cpu().numpy()
                scene_graph = X_gen, Fto_gen, Ffrom_gen
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
            x2 = Xsample + 1
            fto = Fto_vec[:, j - 1] if j > 0 else edge_SOS_token
            ffrom = Ffrom_vec[:, j - 1] if j > 0 else edge_SOS_token

            x1 = model.node_embedding(x1.view(x1.shape[0], 1))
            x2 = model.node_embedding(x2.view(x2.shape[0], 1))
            fto = model.edge_embedding(fto.view(fto.shape[0], 1))
            ffrom = model.edge_embedding(ffrom.view(ffrom.shape[0], 1))
            # scripts gru_edge and sample next edge
            if not model_args.graphrnn_baseline:
                ## fto_out = torch.zeros([1,1,8]).cuda()
                # print(x1.size(), x2.size(), fto.size(), ffrom.size(), fto_out.size())
                ## gru_edge_in1 = torch.cat((x1, x2, fto, ffrom, fto_out), 2)
                gru_edge_in1 = torch.cat((x1, x2, fto, ffrom), 2)
            else:
                gru_edge_in1 = torch.cat((fto, ffrom), 2)
            Fto_scores = model.gru_edge2(gru_edge_in1)
            Fto_scores = torch.squeeze(Fto_scores)
            Fto_sample = torch.multinomial(softmax(Fto_scores), 1)
            Fto_vec[:, j] = torch.squeeze(Fto_sample) + 1
            fto_out = Fto_vec[:, j]
            fto_out = model.edge_embedding(fto_out.view(fto_out.shape[0], 1))
            if not model_args.graphrnn_baseline:
                gru_edge_in2 = torch.cat((x1, x2, fto, ffrom, fto_out), 2)
            else:
                gru_edge_in2 = torch.cat((fto, ffrom), 2)
            Ffrom_scores = model.gru_edge1(gru_edge_in2)
            Ffrom_scores = torch.squeeze(Ffrom_scores)
            Ffrom_sample = torch.multinomial(softmax(Ffrom_scores), 1)
            Ffrom_vec[:, j] = torch.squeeze(Ffrom_sample) + 1
            # update hidden state of gru_edge
            model.gru_edge1.hidden = model.gru_edge1.hidden.data.to(eval_args.device)
            model.gru_edge2.hidden = model.gru_edge2.hidden.data.to(eval_args.device)

        # update hidden state of gru_graph
        model.gru_graph1.hidden = model.gru_graph1.hidden.data.to(eval_args.device)
        model.gru_graph2.hidden = model.gru_graph2.hidden.data.to(eval_args.device)
        model.gru_graph3.hidden = model.gru_graph3.hidden.data.to(eval_args.device)
        Fto[:, :, i + 1] = Fto_vec
        Ffrom[:, :, i + 1] = Ffrom_vec

    return scene_graph


def generate_scene_graphs(
        eval_args: AttributeDict,
        model_args: AttributeDict,
        model: SceneGraphGen,
):
    os.makedirs(eval_args.eval_path, exist_ok=True)
    generated_sg = []

    with open(os.path.join(eval_args.eval_path, 'prior_distribution.p'), 'rb') as f:
        prior = pickle.load(f)

    first_node_list = np.random.choice(list(prior.keys()), eval_args.num_generated_samples,
                                       replace=True, p=list(prior.values()))

    for idx, first_node in enumerate(first_node_list):
        if idx%100==0:
            print('%s iterations done, %s graphs sampled' % (idx, len(generated_sg)))
        scene_graph = generate_one_graph(
            eval_args=eval_args,
            model_args=model_args,
            model=model,
            first_node=first_node
        )
        if scene_graph is not None:
            X, Fto, Ffrom = scene_graph
            F = Fto + np.transpose(Ffrom)
            generated_sg.append([X, F])
    print('sampled %s graphs' % (len(generated_sg)))
    with open(os.path.join(eval_args.eval_path, 'generated_sg.p'), 'wb') as f:
        pickle.dump(generated_sg, f)
