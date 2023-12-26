import math

import yaml


class AttributeDict(dict):
    def __getattr__(self, name):
        return self[name]


def parse_train_args(train_yaml_path):
    with open(train_yaml_path) as f:
        args = yaml.load(f, Loader=yaml.Loader)

    node_decay_steps = int(math.log(args['node_lr_decay']) * args['epochs'] * args['sample_batches'] \
                           / math.log((args['node_lr_end'] / args['node_lr_init'])))
    args['node_step_decay_epochs'] = 1 if node_decay_steps < 1 else node_decay_steps

    edge_decay_steps = int(math.log(args['edge_lr_decay']) * args['epochs'] * args['sample_batches'] \
                           / math.log((args['edge_lr_end'] / args['edge_lr_init'])))
    args['edge_step_decay_epochs'] = 1 if edge_decay_steps < 1 else edge_decay_steps

    # check for missing args
    expected_args = ['data_path', 'model_path', 'device', 'num_workers', 'batch_size', 'sample_batches', 'epochs',
                     'node_lr_init', 'node_lr_end', 'node_lr_decay', 'edge_lr_init', 'edge_lr_end', 'edge_lr_decay',
                     'node_step_decay_epochs', 'edge_step_decay_epochs']
    for arg in expected_args:
        if arg not in args:
            raise ValueError(f'Missing argument {arg} in train yaml file')

    args = AttributeDict(args)
    return args


def parse_model_args(model_yaml_path):
    with open(model_yaml_path) as f:
        args = yaml.load(f, Loader=yaml.Loader)

        args['ggru_input_size'] = 2 * args['edge_emb_size'] * (args['max_edge_len'] - 1) + args['node_emb_size']
        if args['graphrnn_baseline']:
            args['egru_input_size1'] = 2 * args['edge_emb_size']
            args['egru_input_size2'] = 2 * args['edge_emb_size']
        else:
            args['egru_input_size1'] = 3 * args['edge_emb_size'] + 2 * args['node_emb_size']
            args['egru_input_size2'] = 2 * args['edge_emb_size'] + 2 * args['node_emb_size']

    # check for missing args
    expected_args = ['device', 'graphrnn_baseline', 'permutation', 'bias_constant', 'min_num_node', 'max_num_node',
                     'max_edge_len', 'num_node_categories', 'num_edge_categories', 'node_EOS_token', 'edge_SOS_token',
                     'no_edge_token', 'node_emb_size', 'edge_emb_size', 'ggru_emb_size', 'ggru_hidden_size',
                     'ggru_num_layers', 'mlp_input_size', 'mlp_emb_size', 'mlp_out_size', 'egru_hidden_size',
                     'egru_emb_input_size', 'egru_num_layers', 'egru_emb_output_size', 'egru_output_size',
                     'ggru_input_size', 'egru_input_size1', 'egru_input_size2']
    for arg in expected_args:
        if arg not in args:
            raise ValueError(f'Missing argument {arg} in model yaml file')

    args = AttributeDict(args)
    return args


def parse_eval_args(eval_yaml_path):
    with open(eval_yaml_path) as f:
        args = yaml.load(f, Loader=yaml.Loader)

    args = AttributeDict(args)
    return args
