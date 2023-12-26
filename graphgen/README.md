# GraphGen

`DATASET` used below can be any one of `vg`, `vrd`, or `svg`.

## Data Preparation

Run `python make_input_graphs.py --dataset DATASET`

## Model Training

Set parameters in [args.py](args.py). The necessary parameters are:

- `self.note` : decides which baseline to run between GraphGen, GraphRNN and DGMG
- `self.graph_type` : decides which dataset to run the model on

Run `python main.py` to start training. Models will be stored in the location given in `args.py`.

Additional details (not compulsory) can be found in [build.md](build.md).

## Graph Sampling

1. In [evaluate.py](evaluate.py), set parameters:

    - `self.model_path` : location of model saved to be used
    - `self.graphs_save_path` : path (directory) to store all the graphs each in .dat format

2. Run `python evaluate.py`

3. In [combine_generated_graphs.py](combine_generated_graphs.py), set parameters:

    - `graphs_save_path` as the same path used previously in [evaluate.py](evaluate.py)
    - `out_file` as a `.pkl` where all the generated graphs should be saved

4. Run `python combine_generated_graphs.py`. This will save all generated graphs in the `.pkl` file.
