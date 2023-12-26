from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle
import networkx as nx
import json
import pickle
import os
import random
from args import args

#Given id of the object, return the local id i.e in the range (0, num_nodes)
def global_to_local_id(graph, id_):
    for index, node in enumerate(graph.nodes.data()):
        if (node[1]['id'] == id_):
            return index
    return -1


def load_train_data():

    with open('data/scene_graphs.json') as f: sc_gphs = json.load(f)

    graphs = []
    for index_graph , graph in enumerate(sc_gphs):
        G = nx.Graph()
        for index_object, obj in enumerate(graph['objects']): G.add_nodes_from([(index_object, {'label' : obj['name'], 'id' : obj['object_id']})])
        for rel in graph['relationships']: G.add_edge(global_to_local_id(G, rel['subject_id']), global_to_local_id(G, rel['object_id']), label=rel['name'])
        graphs.append(G)

    max_nodes = 0
    for graph in graphs:
        if len(graph.nodes) > max_nodes:
            max_nodes = len(graph.nodes)

    sub_graphs = []
    for graph in graphs:
        sub_graphs += [nx.Graph(graph.subgraph(c)) for c in nx.connected_components(graph) if len(c)>1]

    return sub_graphs

graphs = load_train_data()
random.shuffle(graphs)
graphs = graphs[:110000]

### PARAMETER ###
test_size = args['test_size']
val_size = args['val_size']

graphs_train, graphs_test = train_test_split(graphs, test_size = test_size, random_state=42)
graphs_train, graphs_val = train_test_split(graphs_train, test_size = val_size, random_state=42)
print(len(graphs), len(graphs_train), len(graphs_test), len(graphs_val))

### Init all_stars which is an array of start stars from all graphs
### Init idx2stars that gives an array of next stars from all graphs given a node label and an edge label
def get_all_stars(G):
    stars = []
    nodes = G.nodes
    for node in nodes():
        edges = []
        if G.edges(node):
            for a, b in G.edges(node):
                edges.append(G[a][b]['label'])
            stars.append(nodes[node]['label']+'$%'+'$%'.join(sorted(edges)))
    return stars

def get_star(G, node, label):
    edges = []
    for a, b in G.edges(node):
        edges.append(G[a][b]['label'])
    edges.remove(label)
    return '$%'.join(sorted(edges))

def get_paired_stars(G):
    pairs = []
    for a, b, l in G.edges(data=True):
        pairs.append(G.nodes[a]['label']+'$%'+get_star(G, a, l['label'])+'$$$'+l['label']+'$$$'+G.nodes[b]['label']+'$%'+get_star(G, b, l['label']))
        pairs.append(G.nodes[b]['label']+'$%'+get_star(G, b, l['label'])+'$$$'+l['label']+'$$$'+G.nodes[a]['label']+'$%'+get_star(G, a, l['label']))
    return pairs

all_stars = []
idx2stars = defaultdict(list)
for G in graphs_train:
    all_stars.extend(get_all_stars(G))
    node_labels = G.nodes.data()
    for u, v, l in G.edges(data=True):
        idx2stars[node_labels[v]['label']+'$%'+l['label']].append(node_labels[u]['label'] + '$%' + get_star(G, u, l['label']))
        idx2stars[node_labels[u]['label']+'$%'+l['label']].append(node_labels[v]['label'] + '$%' + get_star(G, v, l['label']))

all_pairs = []
for G in graphs_train: all_pairs.extend(get_paired_stars(G))


### Storing datas needed for this method
with open('data/all_stars.pkl', 'wb') as f: pickle.dump(all_stars, f)
with open('data/all_pair_stars.pkl', 'wb') as f: pickle.dump(all_pairs, f)
with open('data/idx2stars.pkl', 'wb') as f: pickle.dump(idx2stars, f)
with open('data/graphs_train.pkl', 'wb') as f: pickle.dump(graphs_train, f)
with open('data/graphs_test.pkl', 'wb') as f: pickle.dump(graphs_test, f)
with open('data/graphs_val.pkl', 'wb') as f: pickle.dump(graphs_val, f)
