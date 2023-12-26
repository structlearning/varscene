#!/usr/bin/python
import argparse
import os
import pickle
import random

import numpy as np
import torch

from uncond_sggen.utils import load_model
from uncond_sggen.parser import parse_eval_args
from uncond_sggen.evaluate.generate_sg import generate_prior_distribution, generate_scene_graphs
from uncond_sggen.evaluate.mmd_statistics import compute_mmd_statistics

torch.manual_seed(121)
np.random.seed(121)
random.seed(121)

parser = argparse.ArgumentParser()
parser.add_argument('--eval_config', default='../config/args_eval.yaml')
parser.add_argument('--train_config', default='config/args_train.yaml')
parser.add_argument('--model_config', default='config/args_model.yaml')

if __name__ == "__main__":
    args = parser.parse_args()
    eval_args = parse_eval_args(args.eval_config)

    with open(os.path.join(eval_args.data_path, 'train_dataset.p'), 'rb') as f:
        test_dataset = pickle.load(f)

    with open(os.path.join(eval_args.data_path, 'categories.p'), 'rb') as f:
        ind_to_classes, ind_to_predicates, _ = pickle.load(f)

    eval_args.ind_to_classes = ind_to_classes
    eval_args.ind_to_predicates = ind_to_predicates

    model, model_args, train_args = load_model(eval_args, args.train_config, args.model_config)

    # for parameter in model.parameters(): print(parameter)
    print('parameters', sum(p.numel() for p in model.gru_graph1.parameters()))
    print('parameters', sum(p.numel() for p in model.gru_graph2.parameters()))
    print('parameters', sum(p.numel() for p in model.gru_graph3.parameters()))
    print('parameters', sum(p.numel() for p in model.gru_edge1.parameters()))
    print('parameters', sum(p.numel() for p in model.gru_edge2.parameters()))
    print('parameters', sum(p.numel() for p in model.mlp_node.parameters()))

    if eval_args.generate_prior_distribution:
        generate_prior_distribution(
            dataset=test_dataset,
            model_args=model_args,
            eval_args=eval_args
        )
    if eval_args.generate_sg_samples:
        generate_scene_graphs(
            eval_args=eval_args,
            model_args=model_args,
            model=model
        )
    if eval_args.compute_mmd:
        compute_mmd_statistics(
            eval_args=eval_args,
            test_data=test_dataset
        )
