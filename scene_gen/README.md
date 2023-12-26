# SceneGen: Unconditional Scene Graph Generation

Code for the ICCV 2021 paper titled 'Unconditional Scene Graph Generation' (SceneGen)

## Model Training

Run `python main_train.py --train_config PATH_TO_TRAIN_YAML --model_config PATH_TO_MODEL_YAML` (refer key below for the `YAML` paths)

## Graph Sampling

Run:

```bash
python main_eval_sg.py --train_config PATH_TO_TRAIN_YAML --model_config PATH_TO_MODEL_YAML --eval_config PATH_TO_EVAL_YAML
python p_to_nx_graphs.py --dataset DATASET --eval_config PATH_TO_EVAL_YAML
```

This will save generated graphs in `scenegen_generated.pkl` in the `eval_path` specified in the `PATH_TO_EVAL_YAML` config file.

## Key for `YAML` paths

| `DATASET` | `PATH_TO_TRAIN_YAML` | `PATH_TO_MODEL_YAML` | `PATH_TO_EVAL_YAML` |
| ------- | -------------------- | -------------------- | ------------------- |
| `vg` | `config/args_train_vg.yaml` | `config/args_model_vg.yaml` | `config/args_eval_vg.yaml` |
| `vrd` | `config/args_train_vrd.yaml` | `config/args_model_vrd.yaml` | `config/args_eval_vrd.yaml` |
| `svg` | `config/args_train_svg.yaml` | `config/args_model_svg.yaml` | `config/args_eval_svg.yaml` |
