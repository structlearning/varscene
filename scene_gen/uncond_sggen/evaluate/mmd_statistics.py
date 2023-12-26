#!/usr/bin/python
import itertools
import random
from collections import Counter
from typing import List
import os
import torch
import numpy as np
import pickle
from uncond_sggen.parser import AttributeDict


def dirac_set_kernel(Xa, Xb):
    """
    Computes the kernel between two sets of objects (nodes)
    """
    na, ca = Xa
    na = na.long()
    ca = ca.float()
    nodes_a = torch.zeros(150).to(device)
    nodes_a[na] = 1
    counts_a = torch.zeros(150).to(device)
    counts_a[na] = ca
    nb, cb = Xb
    nb = nb.long()
    cb = cb.float()
    nodes_b = torch.zeros(150).to(device)
    nodes_b[nb] = 1
    counts_b = torch.zeros(150).to(device)
    counts_b[nb] = cb
    category_kernel = nodes_a * nodes_b
    count_kernel = 1 / (1 + torch.abs(counts_a - counts_b))

    count_kernel[counts_a == 0] = 0
    count_kernel[counts_b == 0] = 0
    kernel = torch.dot(category_kernel, count_kernel)
    return kernel


def dirac_random_walk_graph_kernel(Ga, Gb, walk_length=3):
    """
    Computes the random walk kernel of two input graphs for a given walk length.
    The node and edge kernels are 1 if categories match otherwise 0.
    """
    Xa, Fa = Ga
    Xb, Fb = Gb
    len_a = Xa.shape[0]
    len_b = Xb.shape[0]
    Xa_count = Counter(list(Xa))
    Xb_count = Counter(list(Xb))
    # init kernel table
    K = np.zeros((walk_length, len_a, len_b))
    # all combinations of node pair
    node_pairs = list(itertools.product(*[np.arange(len_a), np.arange(len_b)]))
    # iteratively estimate kernel for increasing path length
    for p in range(walk_length):
        for pair in node_pairs:
            r = pair[0]
            s = pair[1]
            if Xa[r] == Xb[s]:  # node match
                if p == 0:
                    K[p, r, s] = 1 / (Xa_count[Xa[r]] * Xb_count[Xb[s]])  # set init kernel
                else:
                    # get neighbors of r and s
                    Nr_lst = np.nonzero(Fa[r, :])[0]
                    Ns_lst = np.nonzero(Fb[s, :])[0]
                    # all combinations of node pairs in neighbors
                    N_node_pairs = list(itertools.product(*[Nr_lst, Ns_lst]))
                    # no neighbors of atleast one of r and s
                    if not N_node_pairs:
                        if Nr_lst == [] and Ns_lst == []:
                            neighbor_sim = 1  # when both r and s has no neighbors
                        else:
                            neighbor_sim = 0.5  # when one of r and s has no neighbors
                    else:
                        # when both r and s have neighbors
                        neighbor_sim = 0
                        for N_pair in N_node_pairs:
                            Nr = N_pair[0]
                            Ns = N_pair[1]
                            if Fa[r, Nr] == Fb[s, Ns]:
                                neighbor_sim += K[p - 1, Nr, Ns]
                    # update kernel for current RW of order p
                    K[p, r, s] = K[0, r, s] * neighbor_sim
    kernel = np.sum(K[walk_length - 1])
    return kernel


def dirac_set_kernel_test(Xa, Xb):
    Kab = dirac_set_kernel(Xa, Xb)
    Kaa = dirac_set_kernel(Xa, Xa)
    Kbb = dirac_set_kernel(Xb, Xb)
    return Kab / torch.max(Kaa, Kbb)


def dirac_random_walk_kernel_test(Ga, Gb):
    Kab = dirac_random_walk_graph_kernel(Ga, Gb)
    Kaa = dirac_random_walk_graph_kernel(Ga, Ga)
    Kbb = dirac_random_walk_graph_kernel(Gb, Gb)
    return Kab / max(Kaa, Kbb)


def compute_gram_matrix(graph_data, kernel):
    """
    Computes the NxN normalized gram matrix for the random walk kernel. 
    """
    num_graphs = len(graph_data)
    # init gram matrix
    gram_matrix = np.zeros((num_graphs, num_graphs))
    # get row column idx list
    pair_idx_lst = list(itertools.product(*[np.arange(num_graphs), np.arange(num_graphs)]))
    # compute gram matrix
    for pair_idx in pair_idx_lst:
        a, b = pair_idx
        if kernel=='graph':
            gram_matrix[pair_idx] = dirac_random_walk_graph_kernel(graph_data[a], graph_data[b])
        elif kernel=='node':
            gram_matrix[pair_idx] = dirac_set_kernel(graph_data[a], graph_data[b])
    # normalize gram matrix
    for pair_idx in pair_idx_lst:
        a, b = pair_idx
        if a != b:
            gram_matrix[pair_idx] /= max(gram_matrix[a, a], gram_matrix[b, b])
    # remove diagonal elements
    for idx in range(num_graphs):
        gram_matrix[idx, idx] = 0
    return gram_matrix


def compute_mmd_statistics(
        eval_args: AttributeDict,
        test_data: List
):
    with open(os.path.join(eval_args.eval_path, 'generated_sg.p'), 'r') as f:
        gen_data = pickle.load(f)

    gen_data = random.sample(gen_data, eval_args.num_samples_mmd)
    test_data = random.sample(test_data, eval_args.num_samples_mmd)
    mmd_stat = {}
    for kernel in ['graph', 'set']:
        num_gen = len(gen_data)
        num_data = len(test_data)
        num_all = num_gen + num_data
        all_graphs = gen_data + test_data
        gram_matrix = compute_gram_matrix(all_graphs, kernel)
        total_xx = 1 if num_gen == 1 else num_gen * (num_gen - 1)
        mmd_xx = 1 if num_gen == 1 else np.sum(gram_matrix[0:num_gen, 0:num_gen]) / total_xx

        total_yy = 1 if num_data == 1 else num_data * (num_data - 1)
        mmd_yy = 1 if num_data == 1 else np.sum(gram_matrix[num_gen:num_all, num_gen:num_all]) / total_yy

        mmd_xy = np.sum(gram_matrix[num_gen:num_all, 0:num_gen]) / (num_data * num_gen)

        mmd = mmd_xx + mmd_yy - 2 * mmd_xy
        mmd_stat[kernel] = mmd

    with open(os.path.join(eval_args.eval_path, 'mmd_stats.p'), 'wb') as f:
        pickle.dump(mmd_stat, f)
