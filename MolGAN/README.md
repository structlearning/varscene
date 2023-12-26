# MolGAN

`DATASET` used below can be any one of `vg`, `vrd`, or `svg`.

## Data Preparation

Run `python make_input_graphs.py --dataset DATASET`

## Model Training

Set the 'small_architecture' variable to True in main.py if a model with smaller size is required for large datasets, else by default it is False.

Run `python main.py --dataset DATASET`

## Graph Sampling

Run `python sample.py --dataset DATASET`

This will save generated graphs in `data/DATASET/molgan_generated_graphs.pkl`
