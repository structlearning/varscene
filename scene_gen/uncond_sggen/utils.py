#!/usr/bin/python
import os
from typing import List

import torch
import torch.nn as nn
import yaml
from torch.utils.data import sampler, DataLoader

from uncond_sggen.data import SceneGraphSequenceSampler
from uncond_sggen.model import NodeMLP, GraphGRU, EdgeGRU, SceneGraphGen
from uncond_sggen.parser import AttributeDict
from uncond_sggen.parser import parse_train_args, parse_model_args


def create_data_loader(args: AttributeDict, model_args: AttributeDict, dataset: List):
    train_dataset = SceneGraphSequenceSampler(
        dataset=dataset,
        model_args=model_args
    )
    sample_prob_train = [1.0 / len(train_dataset) for _ in range(len(train_dataset))]
    train_sample_strategy = sampler.WeightedRandomSampler(
        sample_prob_train,
        num_samples=args.sample_batches * args.batch_size,
        replacement=True
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sample_strategy
    )
    return data_loader


def create_model(model_args: AttributeDict):
    # Embeddings for input
    node_emb = nn.Embedding(model_args.num_node_categories,
                            model_args.node_emb_size,
                            padding_idx=0,
                            scale_grad_by_freq=False).to(model_args.device)  # 0, 1to150
    node_emb.weight.requires_grad = False

    edge_emb = nn.Embedding(model_args.num_edge_categories,
                            model_args.edge_emb_size,
                            padding_idx=0,
                            scale_grad_by_freq=False).to(model_args.device)  # 0, 1to50, 51, 52
    edge_emb.weight.requires_grad = False

    # Node Generator
    mlp_node = NodeMLP(model_args=model_args).to(model_args.device)
    gru_graph3 = GraphGRU(model_args=model_args, is_3=True).to(model_args.device)

    # Edge Generator
    gru_graph1 = GraphGRU(model_args=model_args).to(model_args.device)
    gru_graph2 = GraphGRU(model_args=model_args).to(model_args.device)

    gru_edge1 = EdgeGRU(model_args=model_args).to(model_args.device)
    gru_edge2 = EdgeGRU(model_args=model_args, is_2=True).to(model_args.device)

    def num_model_params(model):
        return sum(p.numel() for p in model.parameters())
    print('Individual model parameters:', num_model_params(mlp_node),num_model_params(gru_graph1),
        num_model_params(gru_graph2), num_model_params(gru_graph3), num_model_params(gru_edge1),
        num_model_params(gru_edge2))
    print('Total parameters', num_model_params(mlp_node)+num_model_params(gru_graph1)+num_model_params(gru_graph2)+num_model_params(gru_graph3)+num_model_params(gru_edge1)+num_model_params(gru_edge2))

    return SceneGraphGen(
        node_embedding=node_emb,
        edge_embedding=edge_emb,
        gru_graph1=gru_graph1,
        gru_graph2=gru_graph2,
        gru_graph3=gru_graph3,
        mlp_node=mlp_node,
        gru_edge1=gru_edge1,
        gru_edge2=gru_edge2
    )


def save_model(model: SceneGraphGen, model_args: AttributeDict, train_args: AttributeDict):
    os.makedirs(train_args.model_path, exist_ok=True)
    print('Starting to save model args', os.path.join(train_args.model_path, 'model_args.yaml'))
    # save configs
    with open(os.path.join(train_args.model_path, 'model_args.yaml'), 'w') as f:
        yaml.dump(model_args.__dict__, f, default_flow_style=False)

    print('Starting to save train args', os.path.join(train_args.model_path, 'train_args.yaml'))
    with open(os.path.join(train_args.model_path, 'train_args.yaml'), 'w') as f:
        yaml.dump(train_args.__dict__, f, default_flow_style=False)

    print('Starting to save parameters')
    # save model
    for field in model.__dataclass_fields__:
        print(os.path.join(train_args.model_path, field+'.dat'))
        torch.save(getattr(model, field).state_dict(), os.path.join(train_args.model_path, field+'.dat'))
    print('Saved everything')


def load_model(eval_args: AttributeDict, train_configs, model_configs):
    # load configs
    train_args = parse_train_args(train_configs)
    model_args = parse_model_args(model_configs)
    # with open(model_configs) as f:
    ## with open(os.path.join(eval_args.data_path, "model_args.yaml")) as f:
        # model_args = parse_model_args(yaml.load(f))
    # with open(train_configs) as f:
    ## with open(os.path.join(eval_args.data_path, "train_args.yaml")) as f:
        # train_args = parse_train_args(yaml.load(f))

    # load model
    print(model_args)
    model = create_model(model_args)
    for field in model.__dataclass_fields__:
        module = getattr(model, field)
        module.load_state_dict(
            torch.load(os.path.join(eval_args.model_path, field+'.dat'),
                       map_location=eval_args.device)
        )
        setattr(model, field, module)

    return model, model_args, train_args
