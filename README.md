# VarScene: A Deep Generative Model for Realistic Scene Graph Synthesis (ICML 2022).
Please refer to  [Paper](https://proceedings.mlr.press/v162/verma22b/verma22b.pdf)

## Requirements

- Python 3

## Environment Setup

From the home directory, run:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Use the above virtual environment for training and sampling graphs from all methods.

## Data Preparation

Starting from the home directory, follow these steps for preparing each dataset.

### Visual Genome (vg)

1. Download and unzip `scene_graphs.json` from [here](http://visualgenome.org/static/data/dataset/scene_graphs.json.zip) into the `vg_data/data` directory.
2. Run `cd vg_data`
3. Run `python data_init.py`

### Visual Relationship Detection (vrd)

1. Download `json_dataset` from [here](https://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip), unzip and place all contents of it into the `vrd_data/data` directory.
2. Run `cd vrd_data`
3. Run `python data_init.py`

### Small-sized Visual Genome (svg)

1. Download `train_dataset.p`, `test_dataset.p` and `categories.p` from [here](https://rebrand.ly/varscene) and place it in `svg_data/data`
2. Run `cd svg_data`
3. Run `python data_init.py`

## VarScene

From the `varscene` directory, proceed with following:

`DATASET` below can be any one of `vg`,`svg` or `vrd`

### Model Training

1. `python train_base_model.py --out_dir OUTDIR --dataset DATASET`
2. `python train_property_optim_decoder.py --out_dir OUTDIR --dataset DATASET`: MMD Optimized decoder. This can run only after the `train_base_model.py` script and requires the same `OUTDIR` used in step 1

### Graph Sampling

`OUTDIR` here is the same directory specified in training. The below commands will save a list of generated graphs as `.pkl` files in `OUTDIR`.

- `python unconditional_generation.py --model_dir OUTDIR --dataset DATASET` (unconditional generation)
- `python conditional_generation.py --model_dir OUTDIR --dataset DATASET` (conditional generation)

## Baselines

Instructions for running baselines are present in respective folders. DeepGMG, GraphGen, GraphRNN are in the `graphgen` directory. MolGAN in the `MolGAN` directory and SceneGen in the `SceneGen` directory.

## Evaluation

### Graph Results

From the `varscene` directory, run `python calc_metrics.py --dataset DATASET --data_path PATH_TO_GRAPHS_PKL`

`PATH_TO_GRAPHS_PKL` is the path to the `.pkl` file containing graphs sampled by VarScene or by any other baseline.

This will print metrics for Star-Sim, Edge-sim, Node-sim, SP-K, WL-K, NSPD-K to `STDOUT` (corresponding to table 2 of the paper).

### `sg2im` Results

1. Follow the `sg2im` [README.md](sg2im/README.md) to download the `vg128.pt` model and save it in `sg2im/sg2im-models`
2. From the `directed_varscene` directory, run `python create_sg_json.py --graphs_file PATH_TO_GRAPHS_PKL --out_json_file PATH_TO_SG_JSON` for all `.pkl` files containing sampled scene graphs. This will store the scene graphs in `.json` files required by `sg2im`
3. From the `sg2im` directory, follow its [README.md](sg2im/README.md) to prepare the environment setup and generate scene images from graphs

This completes preparation of all scene images from scene graphs.

The FID, IS, precision and recall values (corresponding to table 5 of the paper) can be computed as follows:

- FID : from the `directed_varscene` directory, run `python fid_score.py PATH_1 PATH_2`. The paths here are directories containing the generated and reference images
- IS : from the `directed_varscene` directory, run `python inception_score.py --path PATH`. `PATH` is the directory containing generated images
- Precision and recall : from the `precision-recall-distributions` directory, follow its [README.md](precision-recall-distributions/README.md) for environment setup. Then run `python prd_from_image_folders.py --reference_dir REF_DIR --eval_dirs EVAL_DIR`

The above commands will all print results to `STDOUT`
