#!/usr/bin/python
import collections
from random import shuffle

import numpy as np

MOTIF_OBJECT_RANK_DICT2 = {
    # PLACE_VEHICLE
    'room': 1, 'tower': 1, 'hill': 1, 'house': 1, 'beach': 1, 'street': 1, 'building': 1, 'mountain': 1, 'sidewalk': 1,
    'track': 1, 'toilet': 1, 'airplane': 1, 'vehicle': 1, 'train': 1, 'bus': 1, 'bike': 1, 'plane': 1, 'truck': 1,
    'motorcycle': 1, 'boat': 1, 'car': 1,
    # PEOPLE_ANIMAL_PLANT_FURNITURE
    'woman': 2, 'lady': 2, 'child': 2, 'guy': 2, 'girl': 2, 'men': 2, 'man': 2, 'people': 2, 'person': 2, 'boy': 2,
    'kid': 2, 'skier': 2, 'player': 2, 'animal': 2, 'bear': 2, 'bird': 2, 'zebra': 2, 'cow': 2, 'cat': 2, 'dog': 2,
    'elephant': 2, 'giraffe': 2, 'sheep': 2, 'horse': 2, 'tree': 2, 'plant': 2, 'basket': 2, 'box': 2, 'cabinet': 2,
    'chair': 2, 'clock': 2, 'counter': 2, 'curtain': 2, 'desk': 2, 'door': 2, 'seat': 2, 'shelf': 2, 'fence': 2,
    'board': 2, 'lamp': 2, 'vase': 2, 'sink': 2, 'bench': 2, 'bed': 2, 'stand': 2, 'table': 2, 'drawer': 2,
    'pillow': 2, 'light': 2, 'tile': 2, 'railing': 2, 'roof': 2, 'glass': 2, 'bottle': 2, 'bowl': 2, 'cup': 2,
    'plate': 2, 'fork': 2, 'pot': 2, 'banana': 2, 'pizza': 2, 'vegetable': 2, 'orange': 2, 'food': 2, 'fruit': 2,
    # PARTOF PEOPLE_ANIMAL_PLANT_FURNITURE_VEHICLE
    'wheel': 3, 'window': 3, 'windshield': 3, 'handle': 3, 'engine': 3, 'tire': 3, 'pole': 3, 'post': 3, 'snow': 3,
    'wave': 3, 'wire': 3, 'flag': 3, 'kite': 3, 'rock': 3, 'letter': 3, 'logo': 3, 'paper': 3, 'number': 3,
    'screen': 3, 'sign': 3, 'eye': 3, 'face': 3, 'ear': 3, 'hair': 3, 'hand': 3, 'head': 3, 'arm': 3, 'leg': 3,
    'wing': 3, 'mouth': 3, 'neck': 3, 'nose': 3, 'paw': 3, 'branch': 3, 'finger': 3, 'flower': 3, 'leaf': 3,
    'trunk': 3, 'tail': 3, 'hat': 3, 'cap': 3, 'coat': 3, 'helmet': 3, 'jacket': 3, 'jean': 3, 'glove': 3, 'shirt': 3,
    'shoe': 3, 'short': 3, 'tie': 3, 'pant': 3, 'boot': 3, 'sock': 3, 'skateboard': 3, 'ski': 3, 'sneaker': 3,
    'surfboard': 3, 'racket': 3, 'bag': 3, 'book': 3, 'laptop': 3, 'phone': 3, 'towel': 3, 'umbrella': 3
}


def motif_based_ordered(node_x, edge_f, ind_to_classes):
    obj_rank = [[idx, MOTIF_OBJECT_RANK_DICT2[ind_to_classes[o - 1]]] for idx, o in enumerate(node_x)]
    sorted_pair = np.array(sorted(obj_rank, key=lambda x: x[1]))
    order_idx = [int(i) for i in sorted_pair[:, 0]]
    node_x = node_x[order_idx]  # 0 to 149
    edge_f = edge_f[np.ix_(order_idx, order_idx)]  # 1 to 50
    return node_x, edge_f


def random_ordered(node_x, edge_f):
    order_idx = list(range(node_x.shape[0]))
    shuffle(order_idx)
    node_x = node_x[order_idx]  # 0 to 149
    edge_f = edge_f[np.ix_(order_idx, order_idx)]  # 1 to 50
    return node_x, edge_f


def predefined_ordered(node_x, edge_f):
    node_x, edge_f = random_ordered(node_x, edge_f)
    obj_idx_pair = [[idx, obj] for idx, obj in enumerate(node_x)]
    sorted_pair = np.array(sorted(obj_idx_pair, key=lambda x: x[1]))
    order_idx = [int(i) for i in sorted_pair[:, 0]]
    node_x = node_x[order_idx]  # 0 to 149
    edge_f = edge_f[np.ix_(order_idx, order_idx)]  # 1 to 50
    return node_x, edge_f


def bfs_ordered(node_x, edge_f, root):
    graph = {}
    for i in range(node_x.shape[0]):
        edges = set(np.nonzero(edge_f[:, i])[0])
        edges.update(set(np.nonzero(edge_f[i, :])[0]))
        edges = list(edges)
        if edges:
            shuffle(edges)
        graph[i] = edges
    order_idx = np.array(breadth_first_search(graph, root))
    node_x = node_x[order_idx]  # 0 to 149
    edge_f = edge_f[np.ix_(order_idx, order_idx)]  # 1 to 50
    return node_x, edge_f


def breadth_first_search(graph, root):
    order = list()
    visited, queue = set(), collections.deque([root])
    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        visited.add(vertex)
        queue.extend(n for n in graph[vertex] if n not in visited and n not in queue)

    remaining = [node for node in graph if node not in order]
    if remaining:
        graph = {k: v for k, v in graph.items() if k in remaining}
        shuffle(remaining)
        root = remaining[0]
        order.extend(breadth_first_search(graph, root))
    return order


def get_dist_connection_range(graphs_all, threshold, iterations):
    connection_range = list()
    outlier_graphs = set()
    for it in range(iterations):
        for idx, graph in enumerate(graphs_all):
            X, F = graph
            X, F = bfs_ordered(X, F)
            Fto = np.triu(F, +1)
            Ffrom = np.transpose(np.tril(F, -1))
            for i in range(X.shape[0]):
                edge_idx = list(np.nonzero(Fto[:, i])[0])
                if edge_idx:
                    dist = i - 1 - edge_idx[0]
                    if dist != 0:
                        connection_range.append(dist)
                    if dist > threshold:
                        outlier_graphs.add(idx)

                edge_idx = list(np.nonzero(Ffrom[:, i])[0])
                if edge_idx:
                    dist = i - 1 - edge_idx[0]
                    if dist != 0:
                        connection_range.append(dist)
                    if dist > threshold:
                        outlier_graphs.add(idx)

    return connection_range, outlier_graphs
