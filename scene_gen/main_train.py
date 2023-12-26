#!/usr/bin/python
import argparse
import os
import pickle
import random

import numpy as np
import torch

from uncond_sggen.utils import create_model, create_data_loader
from uncond_sggen.parser import parse_train_args, parse_model_args
from uncond_sggen.trainer import SceneGraphGenTrainer

torch.manual_seed(121)
np.random.seed(121)
random.seed(121)

parser = argparse.ArgumentParser()
parser.add_argument('--train_config', default='config/args_train.yaml')
parser.add_argument('--model_config', default='config/args_model.yaml')


if __name__ == "__main__":
    args = parser.parse_args()
    train_args = parse_train_args(args.train_config)
    model_args = parse_model_args(args.model_config)

    with open(os.path.join(train_args.data_path, 'train_dataset.p'), 'rb') as f:
        train_dataset = pickle.load(f)

    model = create_model(model_args)
    data_loader = create_data_loader(
        args=train_args,
        model_args=model_args,
        dataset=train_dataset
    )
    trainer = SceneGraphGenTrainer(
        train_args=train_args,
        model_args=model_args,
        data_loader=data_loader,
        model=model
    )
    trainer.train()
