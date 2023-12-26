from utils import nx_to_json, graph_in_vocab
import torch
import pickle
import networkx as nx
import argparse

## load vocabulary
sg2im_checkpoint = torch.load('../sg2im/sg2im-models/vg128.pt', map_location='cpu')
vocab = sg2im_checkpoint['model_kwargs']['vocab']
object_vocab = vocab['object_name_to_idx']
pred_vocab = vocab['pred_name_to_idx']

def create_json(nx_file, json_file):
    with open(nx_file, 'rb') as f:
        gph_set = pickle.load(f)
    if 'molgan' in nx_file:
        new_gph_set = []
        for g in gph_set:
            good_nodes = [u for u, l in g.nodes(data='label') if l != 'NO']
            new_gph_set.append(nx.subgraph(g ,good_nodes))
        gph_set = new_gph_set
    gphs = [nx.convert_node_labels_to_integers(nx.DiGraph(g)) for g in gph_set if graph_in_vocab(g, object_vocab, pred_vocab)]
    print('%s/%s in vocab' % (len(gphs), len(gph_set)))
    nx_to_json(gphs, json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs_file', type=str, required=True, help='path of pkl file of graphs')
    parser.add_argument('--out_json_file', type=str, required=True, help='path of final json file')
    args = parser.parse_args()
    create_json(args.graphs_file, args.out_json_file)


