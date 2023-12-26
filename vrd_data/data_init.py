import networkx as nx
import json
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

## load data from json
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

annotations_train = load_json('annotations_train.json')
annotations_test = load_json('annotations_test.json')
objects = load_json('objects.json')
predicates = load_json('predicates.json')

graphs_train, graphs_test = [], []

def create_graph_list(graph_dict):
    graph_list = []
    num_cyclic = 0
    for k,v in graph_dict.items():

        G = nx.Graph()
        node_id_dict = dict() ## node id corresponding to (bbox, category)
        node_id = 0

        for edge in v:

            edge_label = predicates[edge['predicate']]
            ## use (bbox, category) as id of nodes
            obj = edge['object']
            obj_id = tuple(obj['bbox']+[obj['category']])
            sub = edge['subject']
            sub_id = tuple(sub['bbox']+[sub['category']])

            if obj_id not in node_id_dict:
                node_id_dict[obj_id] = node_id
                node_id += 1
            if sub_id not in node_id_dict:
                node_id_dict[sub_id] = node_id
                node_id += 1
            obj_id, sub_id = node_id_dict[obj_id], node_id_dict[sub_id]

            G.add_edge(obj_id, sub_id, label=edge_label)
            
            G.nodes[obj_id]['label'] = objects[obj['category']]
            G.nodes[sub_id]['label'] = objects[sub['category']]

        if nx.number_of_nodes(G) == 0:
            continue

        connected_components = [G.subgraph(c).copy() for c in nx.connected_components(G) if len(c)>1]

        graph_list += connected_components

    print('total %s' % len(graph_list))

    return graph_list

graphs_test = create_graph_list(annotations_test)
graphs_val1, graphs_test = train_test_split(graphs_test, test_size=1000, random_state=42)
graphs_train = create_graph_list(annotations_train)
graphs_train, graphs_val2 = train_test_split(graphs_train, test_size=1000-len(graphs_val1), random_state=42)
graphs_val = graphs_val1 + graphs_val2

print('train: %s, val: %s, test: %s graphs' % (len(graphs_train), len(graphs_val), len(graphs_test)))

## create star data
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
