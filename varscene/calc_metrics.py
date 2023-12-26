from scipy.linalg import eigvalsh
import argparse
import numpy as np
from time import time
from collections import Counter
import networkx as nx
import pickle
import random
import os
import subprocess as sp
from grakel.graph_kernels import ShortestPath, NeighborhoodSubgraphPairwiseDistance, EdgeHistogram, WeisfeilerLehman
import grakel.utils
import itertools
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as nx_gph_hash

np.random.seed(1)
random.seed(2)

def get_all_stars(G):
    stars = []
    for node, node_label in G.nodes('label', default='__dummy__'):
        edges = []
        for _, _, e_label in G.edges(node, 'label', default='__dummy__'):
            edges.append(e_label)
        stars.append(tuple([node_label] + sorted(edges)))
    return stars

def gaussian(x, y, sigma=1.0):  
	support_size = max(len(x), len(y))
  	# convert histogram values x and y to float, and make them equal len
	x = x.astype(float)
	y = y.astype(float)
	if len(x) < len(y):
		x = np.hstack((x, [0.0] * (support_size - len(x))))
	elif len(y) < len(x):
		y = np.hstack((y, [0.0] * (support_size - len(y))))

	dist = np.linalg.norm(x - y, 2)
	return np.exp(-dist * dist / (2 * sigma * sigma))

def get_all_stars_mmd_opt(G):
    stars = []
    for node, node_label in G.nodes('label', default='__dummy__'):
        edges = []
        if G.edges(node):
            for _, _, e_label in G.edges(node, 'label', default='__dummy__'):
                edges.append(e_label)
            stars.append(node_label+'$%'+'$%'.join(sorted(edges)))
    return stars

### single star dist ###
def single_star_worker_mmd_opt(G, sampled_stars2idx):
    stars = get_all_stars_mmd_opt(G)
    star_counts = Counter(stars)
    out = np.zeros(len(sampled_stars2idx), dtype = np.int8)
    for s in star_counts:
        if s in sampled_stars2idx: out[sampled_stars2idx[s]] += star_counts[s]
    return out

def get_star(G, node, label):
    edges = list()
    for _, _, e_label in G.edges(node, 'label', default='__dummy__'):
        edges.append(e_label)
    edges.remove(label)
    return '$%'.join(sorted(edges))

def get_paired_stars(G):
    pairs = []
    for a, b, l in G.edges(data='label', default='__dummy__'):
        a_data = G.nodes[a]
        b_data = G.nodes[b]
        a_label = a_data['label'] if 'label' in a_data else '__dummy__'
        b_label = b_data['label'] if 'label' in b_data else '__dummy__'
        pairs.append(a_label+'$%'+get_star(G, a, l)+'$$$'+l+'$$$'+b_label+'$%'+get_star(G, b, l))
        pairs.append(b_label+'$%'+get_star(G, b, l)+'$$$'+l+'$$$'+a_label+'$%'+get_star(G, a, l))
    return pairs

def get_node_unigrams(G):
    node_unigrams = list()
    for _, label in G.nodes(data='label', default='__dummy__'):
        node_unigrams.append(label)
    return node_unigrams

def get_node_bigrams(G):
    node_bigrams = list()
    for u, v, e_label in G.edges(data='label', default='__dummy__'):
        u_label = G.nodes[u]['label'] if 'label' in G.nodes[u] else '__dummy__'
        v_label = G.nodes[v]['label'] if 'label' in G.nodes[v] else '__dummy__'
        node_bigrams.append(tuple([e_label]+sorted([u_label, v_label])))
    return node_bigrams

def get_node_trigrams(G):
    node_trigrams = list()
    for node, node_label in G.nodes(data='label', default='__dummy__'):
        incident_edges = [(v, e_label) for _, v, e_label in G.edges(node, data='label', default='__dummy__')]
        for i in range(len(incident_edges)):
            for j in range(i+1, len(incident_edges)):
                v, e_label = incident_edges[i]
                e1 = (G.nodes[v]['label'] if 'label' in G.nodes[v] else '__dummy__', e_label)
                v, e_label = incident_edges[j]
                e2 = (G.nodes[v]['label'] if 'label' in G.nodes[v] else '__dummy__', e_label)
                node_trigrams.append(tuple([node_label]+sorted([e1, e2])))
    return node_trigrams

def get_edge_unigrams(G):
    edge_unigrams = list()
    for _, _, e_label in G.edges(data='label', default='__dummy__'):
        edge_unigrams.append(e_label)
    return edge_unigrams

def get_edge_bigrams(G):
    edge_bigrams = list()
    for node, node_label in G.nodes(data='label', default='__dummy__'):
        incident_edges = [e_label for _, _, e_label in G.edges(node, data='label', default='__dummy__')]
        for i in range(len(incident_edges)):
            for j in range(i+1, len(incident_edges)):
                e1 = incident_edges[i]
                e2 = incident_edges[j]
                if ('__dummy' in e1 and '__dummy' in e2) or \
                    ('__dummy' not in e1 and '__dummy' not in e2):
                    edge_bigrams.append(tuple([node_label]+sorted([e1, e2])))
    return edge_bigrams

def get_edge_trigrams(G):
    edge_trigrams = list()
    for u, v, e_label in G.edges(data='label', default='__dummy__'):
        
        u_label = G.nodes[u]['label'] if 'label' in G.nodes[u] else '__dummy__'
        u_edges = [e_label for _, _, e_label in G.edges(u, data='label', default='__dummy__')]
        u_edges.remove(e_label)

        v_label = G.nodes[v]['label'] if 'label' in G.nodes[v] else '__dummy__'
        v_edges = [e_label for _, _, e_label in G.edges(v, data='label', default='__dummy__')]
        v_edges.remove(e_label)

        for e1 in u_edges:
            for e2 in v_edges:
                node_edge1 = (u_label, e1)
                node_edge2 = (v_label, e2)
                edge_trigrams.append(tuple([e_label]+sorted([node_edge1, node_edge2])))

    return edge_trigrams

def get_subgraph_hashes(G, size=3):
    ## return list of all induced subgraphs of given size
    g = nx.convert_node_labels_to_integers(G)
    subgraph_nodes = itertools.combinations(range(g.number_of_nodes()), size)
    subgraphs = list()
    for subg in subgraph_nodes:
        sg = nx.subgraph(nx.Graph(g), subg)
        subgraphs.append(nx_gph_hash(sg, 'label', 'label'))
    return subgraphs

def get_open_edge_subgraph_hashes(G, size=3):
    g = nx.convert_node_labels_to_integers(nx.Graph(G))
    subgraph_nodes = itertools.combinations(range(g.number_of_nodes()), size)
    subgraphs = list()
    for subg in subgraph_nodes:
        sg = nx.Graph()
        for node in subg:
            node_label = g.nodes[node]['label'] if 'label' in g.nodes[node] else '__dummy__'
            sg.add_node(node, label=node_label)
        for node in subg:
            for _, v, e_label in g.edges(node, data='label'):
                v_node_num = v
                if v not in subg:
                    v_node_num = g.number_of_nodes() + sg.number_of_nodes()
                    sg.add_node(v_node_num, label='__dummy__')
                sg.add_edge(node, v_node_num, label=e_label)
        subgraphs.append(nx_gph_hash(sg, 'label', 'label'))
    return subgraphs

def disc(samples1, samples2, kernel, *args, **kwargs):
    d = 0
    for s1 in samples1:
        for s2 in samples2:
            d += kernel(s1, s2, *args, **kwargs)
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, *args, **kwargs):
    s11 = disc(samples1, samples1, kernel, *args, **kwargs)
    s22 = disc(samples2, samples2, kernel, *args, **kwargs)
    s12 = disc(samples1, samples2, kernel, *args, **kwargs)
    return np.sqrt(s11 + s22 - 2 * s12)

def cos_sim(v1, v2):
    max_sz = max(len(v1), len(v2))
    v1 = np.concatenate((v1, np.zeros(max_sz-len(v1))))
    v2 = np.concatenate((v2, np.zeros(max_sz-len(v2))))
    # return np.linalg.norm(v1-v2, ord=1)
    if np.sum(v1*v2) != 0:
        return np.sum(v1*v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return 0.0

def make_dist(gphs, worker, normalize=False, **kwargs):
    return worker(gphs, **kwargs)

def make_dist_mmd_opt(gphs, worker, normalize=True, **kwargs):
    out = list()
    for g in gphs:
        distribution = np.array(worker(g, **kwargs), dtype=np.float16)
        if normalize:
            if np.sum(distribution) > 0: ## normalize
                distribution /= np.sum(distribution)
        out.append(distribution)
    return out


def graph_structure_worker(gphs, structure2idx, get_func):
    out = np.zeros(len(structure2idx))
    for g in gphs:
        structures = get_func(g)
        structure_counts = Counter(structures)
        for s in structure_counts:
            out[structure2idx[s]] += structure_counts[s]
    return out / len(gphs)

### single star dist ###
def single_star_worker(gphs, sampled_stars2idx):
    out = np.zeros(len(sampled_stars2idx))
    for g in gphs:
        stars = get_all_stars(g)
        star_counts = Counter(stars)
        for s in star_counts:
            out[sampled_stars2idx[s]] += star_counts[s]
    return out / len(gphs)

def novelty(gphs_pred, gphs_ref):
    ref_hash_set = set()
    for g in gphs_ref:
        ref_hash_set.add(nx_gph_hash(nx.Graph(g), 'label', 'label'))
    cnt_common = 0
    for g in add_dummy_labels(gphs_pred, False):
        g_hash = nx_gph_hash(nx.Graph(g), 'label', 'label')
        if g_hash in ref_hash_set:
            cnt_common += 1
    return 1 - cnt_common/len(gphs_pred)

def uniqueness(gphs):
    hash_set = set()
    for g in add_dummy_labels(gphs, False):
        hash_set.add(nx_gph_hash(nx.Graph(g), 'label', 'label'))
    return len(hash_set)/len(gphs)

def add_dummy_labels(gphs, atleast_one_edge=True):
    complete_label_gphs = list()
    for g in gphs:
        g_copy = nx.convert_node_labels_to_integers(nx.Graph(g))
        nodes = list(g_copy.nodes())
        for u in nodes:
            if 'label' not in g_copy.nodes[u]:
                g_copy.nodes[u]['label'] = '__dummy__'
        edges = list(g_copy.edges())
        if atleast_one_edge and len(edges) == 0:
            g_copy.add_edge(0, 1, label='__dummy__')
        for u, v in edges:
            if 'label' not in g_copy.edges[u,v]:
                g_copy.edges[u,v]['label'] = '__dummy__'
        complete_label_gphs.append(g_copy)
    return complete_label_gphs

def add_dummy_edges(G, unique_dummy_edge=True):
    newG = nx.convert_node_labels_to_integers(G)
    num_nodes = newG.number_of_nodes()
    isolated_nodes = []
    for u in range(num_nodes):
        if len(newG.edges(u)) == 0:
            isolated_nodes.append(u)
    for u in isolated_nodes:
        # newG.nodes[u]['label'] = newG.nodes[u]['label'] + '__isolated__'
        for v in range(num_nodes):
            if v == u or (u,v) in newG.edges(G):
                continue ## don't add self loop or extra edge
            if unique_dummy_edge:
                edge_label = '__dummy__'
            else:
                edge_label = '__dummy_%.20f__' % np.random.random()
            newG.add_edge(u, v, label=edge_label)
    return newG

grkl_kernels = dict(sp=None, nspd=None, edge_hist=None, WL=None,
    sp_norm=None, nspd_norm=None, edge_hist_norm=None, WL_norm=None)
def compute_grkl_kernel(gphs_pred, gphs_ref, kernel, normalize=True):
    if kernel not in ['sp', 'nspd', 'edge_hist', 'WL',\
        'sp_norm', 'nspd_norm', 'edge_hist_norm', 'WL_norm']:
        raise ValueError('invalid grakel kernel')
    
    if grkl_kernels[kernel] is None:
        grkl_gphs = add_dummy_labels(gphs_ref)
        grkl_gphs = grakel.utils.graph_from_networkx(grkl_gphs, 'label', 'label')
        if kernel.startswith('sp'):
            grkl_kernels[kernel] = ShortestPath(normalize=normalize)
        elif kernel.startswith('nspd'):
            grkl_kernels[kernel] = NeighborhoodSubgraphPairwiseDistance(normalize=normalize)
        elif kernel.startswith('edge_hist'):
            grkl_kernels[kernel] = EdgeHistogram(normalize=normalize)
        elif kernel.startswith('WL'):
            grkl_kernels[kernel] = WeisfeilerLehman(normalize=normalize)
        grkl_kernels[kernel].fit_transform(grkl_gphs)

    grkl_gphs = [o for o in gphs_pred if o.number_of_nodes()>1]
    grkl_gphs = add_dummy_labels(grkl_gphs)
    grkl_gphs = grakel.utils.graph_from_networkx(grkl_gphs, 'label', 'label')
    K_pred = grkl_kernels[kernel].transform(grkl_gphs)
    K_pred = np.nan_to_num(K_pred)
    return np.mean(K_pred)

grkl_kernels_dist = dict(sp=None, nspd=None, edge_hist=None, WL=None,
    sp_norm=None, nspd_norm=None, edge_hist_norm=None, WL_norm=None)
K_ref = None
def compute_grkl_kernel_dist(gphs_pred, gphs_ref, kernel, normalize=True):
    if kernel not in ['sp', 'nspd', 'edge_hist', 'WL',\
        'sp_norm', 'nspd_norm', 'edge_hist_norm', 'WL_norm']:
        raise ValueError('invalid grakel kernel')
    
    if grkl_kernels_dist[kernel] is None:
        grkl_gphs = add_dummy_labels(gphs_ref)
        grkl_gphs = grakel.utils.graph_from_networkx(grkl_gphs, 'label', 'label')
        if kernel.startswith('sp'):
            grkl_kernels_dist[kernel] = ShortestPath(normalize=normalize)
        elif kernel.startswith('nspd'):
            grkl_kernels_dist[kernel] = NeighborhoodSubgraphPairwiseDistance(normalize=normalize)
        elif kernel.startswith('edge_hist'):
            grkl_kernels_dist[kernel] = EdgeHistogram(normalize=normalize)
        elif kernel.startswith('WL'):
            grkl_kernels_dist[kernel] = WeisfeilerLehman(normalize=normalize)
        global K_ref
        K_ref = grkl_kernels_dist[kernel].fit_transform(grkl_gphs)
        K_ref = np.mean(np.nan_to_num(K_ref))

    dist = K_ref

    grkl_gphs = add_dummy_labels(gphs_pred)
    grkl_gphs = [o for o in grkl_gphs if o.number_of_edges()>0]
    grkl_gphs = list(grakel.utils.graph_from_networkx(grkl_gphs, 'label', 'label'))
    K_pred_ref = grkl_kernels_dist[kernel].transform(grkl_gphs)
    K_pred_ref = np.mean(np.nan_to_num(K_pred_ref))
    dist -= 2*K_pred_ref

    if kernel.startswith('sp'):
        kernel_copy = ShortestPath(normalize=normalize)
    elif kernel.startswith('nspd'):
        kernel_copy = NeighborhoodSubgraphPairwiseDistance(normalize=normalize)
    elif kernel.startswith('edge_hist'):
        kernel_copy = EdgeHistogram(normalize=normalize)
    elif kernel.startswith('WL'):
        kernel_copy = WeisfeilerLehman(normalize=normalize)
    K_pred = kernel_copy.fit_transform(grkl_gphs)
    K_pred = np.mean(np.nan_to_num(K_pred))
    dist += K_pred

    return dist

def compute_all_metrics(graphs_pred, graphs_ref, metrics, name='no_name'):

    if metrics is None:
        metrics = [
            'gk_sp', 'gk_WL', 'gk_nspd', 'single_star', 'edge_bigrams', 'node_bigrams'
        ]
    results = dict()

    for worker in ['node_unigrams', 'node_bigrams', 'node_trigrams', 'edge_unigrams', 'edge_bigrams', 'edge_trigrams']:
        if worker in metrics:
            t1 = time()
            all_strucs = list()
            for g in graphs_pred+graphs_ref:
                all_strucs.extend(eval('get_%s' % worker)(g))
            all_strucs = list(set(all_strucs))
            idx2strucs = all_strucs
            struc2idx = {}
            for i, e in enumerate(idx2strucs): struc2idx[e] = i
            dist_1 = graph_structure_worker(graphs_pred, struc2idx, eval('get_%s' % worker))
            dist_2 = graph_structure_worker(graphs_ref, struc2idx, eval('get_%s' % worker))
            results[worker] = cos_sim(dist_1, dist_2)
            print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))
    
    if 'size' in metrics:
        t1 = time()
        worker = 'size'
        results[worker] = len(graphs_pred)
        print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))

    ## single star
    if 'single_star' in metrics:
        t1 = time()
        ref_stars, extra_stars, all_stars = [], [], []
        
        for g in graphs_ref:
            ref_stars.extend(get_all_stars(g))
        ref_stars = set(ref_stars)

        for g in graphs_pred:
            extra_stars.extend(get_all_stars(g))
        extra_stars = set(extra_stars).difference(ref_stars)

        ref_stars = list(ref_stars)
        extra_stars = list(extra_stars)

        all_stars = ref_stars + extra_stars

        idx2sampled_stars = all_stars
        stars2idx = {}
        for i, e in enumerate(idx2sampled_stars): stars2idx[e] = i
        worker = 'single_star'
        dist_1 = make_dist(graphs_pred, eval(worker+'_worker'), sampled_stars2idx=stars2idx)
        dist_2 = make_dist(graphs_ref, eval(worker+'_worker'), sampled_stars2idx=stars2idx)
        results[worker] = cos_sim(dist_1, dist_2)
        print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))

    ## novelty
    if 'novelty' in metrics:
        t1 = time()
        worker = 'novelty'
        results[worker] = novelty(graphs_pred, graphs_ref)
        print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))
    
    ## uniqueness
    if 'uniqueness' in metrics:
        t1 = time()
        worker = 'uniqueness'
        results[worker] = uniqueness(graphs_pred)
        print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))

    if 'gk_sp' in metrics:
        t1 = time()
        worker = 'gk_sp'
        results[worker] = compute_grkl_kernel(graphs_pred, graphs_ref, worker[3:], False)
        print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))
    
    if 'gk_nspd' in metrics:
        t1 = time()
        worker = 'gk_nspd'
        results[worker] = compute_grkl_kernel(graphs_pred, graphs_ref, worker[3:], False)
        print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))
    
    if 'gk_WL' in metrics:
        t1 = time()
        worker = 'gk_WL'
        results[worker] = compute_grkl_kernel(graphs_pred, graphs_ref, worker[3:], False)
        print('%s data, %s = %.7f, computed in %.2fs' % (name, worker, results[worker], time()-t1))
    #########################################################

    print('%s graphs' % name)
    for metric in metrics:
        if type(results[metric]) == tuple:
            print('%s %.7f %.7f' % (metric, *results[metric]))
        else:
            print('%s %.7f' % (metric, results[metric]))

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['vrd', 'vg', 'svg'], help='dataset name', type=str, required=True)
    parser.add_argument('--data_path', required=True, help='pkl file path of graphs', type=str)
    parser.add_argument('--ref_gphs', help='dataset against which metrics are to be computed', type=str, choices=['train', 'val', 'train_val', 'test'], default='test')
    args = parser.parse_args()

    ### Read All Datasets ###
    if args.dataset == 'vrd':
        data_dir = '../vrd_data/data'
        num_graphs = 5000
    elif args.dataset == 'vg':
        data_dir = '../vg_data/data'
        num_graphs = 9950
    elif args.dataset == 'svg':
        data_dir = '../svg_data/data'
        num_graphs = 10000
    t1 = time()
    with open(os.path.join(data_dir, 'graphs_train.pkl'), 'rb') as f: graphs_train = pickle.load(f)
    with open(os.path.join(data_dir, 'graphs_test.pkl'), 'rb') as f: graphs_test = pickle.load(f)
    with open(os.path.join(data_dir, 'graphs_val.pkl'), 'rb') as f: graphs_val = pickle.load(f)
    print('loaded datasets in %.2fs' % (time()-t1))
    print('%s train, %s val, %s test graphs' % (len(graphs_train), len(graphs_val), len(graphs_test)))

    graphs_ref = eval('graphs_'+args.ref_gphs) if args.ref_gphs != 'train_val' else graphs_train+graphs_val
    metrics = ['single_star', 'edge_bigrams', 'node_bigrams', 'gk_sp', 'gk_WL', 'gk_nspd']
    kwargs = dict(graphs_ref=graphs_ref, metrics=metrics)
    results = eval(args.dataset + '_runner')(graphs_val, graphs_train, **kwargs)

    with open(args.data_path, 'rb') as f: graphs = pickle.load(f)
    results[args.data_path] = compute_all_metrics(graphs[:num_graphs], name=args.data_path, **kwargs)

    ## print results
    for k, v in results.items():
        print('model', end=' ')
        for m in metrics:
            print('& %s' % m, end=' ')
        print()
        break
    
