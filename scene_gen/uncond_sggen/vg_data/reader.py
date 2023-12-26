#!/usr/bin/python
import json

import h5py
import numpy as np


# ----------Load and pre-process graphs----------------------------
def item_to_index(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def clean_graph(obj_pair_id, relations):
    # multiple identical triplets are reduced to one.
    # multiple edges from one node to another is reduced to one
    obj_pair_lst = [tuple(row) for row in obj_pair_id]
    duplicate_id = [item_to_index(obj_pair_lst, x) for x in set(obj_pair_lst)
                    if obj_pair_lst.count(x) > 1]
    id_to_delete = list()
    for dup_ids in duplicate_id:
        id_to_delete.extend(dup_ids[1:])

    obj_pair_id = np.delete(obj_pair_id, id_to_delete, axis=0)
    relations = np.delete(relations, id_to_delete, axis=0)

    return obj_pair_id, relations


def load_scene_graphs(graphs_file, num_graphs=None):
    # Read graph file
    roi_h5 = h5py.File(graphs_file, 'r')

    # Load contents from file
    all_relations = roi_h5['predicates'][:, 0]
    all_objects = roi_h5['labels'][:, 0]
    all_obj_pair_id = roi_h5['relationships']
    assert (all_obj_pair_id.shape[0] == all_relations.shape[0])

    im_to_first_rel = roi_h5['img_to_first_rel']
    im_to_last_rel = roi_h5['img_to_last_rel']
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])

    all_boxes = roi_h5['boxes_512'][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box
    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box']
    im_to_last_box = roi_h5['img_to_last_box']
    assert (im_to_first_box.shape[0] == im_to_last_box.shape[0])

    # segregate graphs in a list
    graph_list = list()
    if num_graphs == -1:
        num_graphs = len(im_to_last_rel)
        idx_list = np.arange(len(im_to_last_rel))
    else:
        idx_list = np.random.choice(np.arange(len(im_to_last_rel)), num_graphs,
                                    replace=False)

    for i in idx_list:
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_objects[im_to_first_box[i]:im_to_last_box[i] + 1] - 1
        obj_pair_id = all_obj_pair_id[im_to_first_rel[i]: im_to_last_rel[i] + 1, :]
        relations = all_relations[im_to_first_rel[i]: im_to_last_rel[i] + 1] - 1

        if relations.shape[0] != 0:
            obj_pair_id, relations = clean_graph(obj_pair_id, relations)
            obj_pair = np.column_stack((all_objects[obj_pair_id[:, 0]] - 1,
                                        all_objects[obj_pair_id[:, 1]] - 1))
            triplets = np.column_stack((obj_pair, relations))

            graph = {'obj_pair_id': obj_pair_id,
                     'triplets': triplets,
                     'objects': gt_classes_i,
                     'boxes': boxes_i,
                     }

            # map objects to dict with index of occurence of that object
            obj_to_idx = dict((x, item_to_index(gt_classes_i, x)) for i, x in enumerate(set(gt_classes_i)))
            # map obj_ids of all triplets to those unique ids 
            triplet_obj_ids = np.unique(obj_pair_id)
            triplet_to_object_map = dict()
            for obj_id in triplet_obj_ids:
                obj = all_objects[obj_id] - 1
                triplet_to_object_map[obj_id] = obj_to_idx[obj].pop(0)

            graph['triplet_to_object_map'] = triplet_to_object_map
            graph_list.append(graph)

    return graph_list


def get_vocab(info_file):
    # Read graph vocabulary
    info = json.load(open(info_file, 'r'))
    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']

    ind_to_classes = np.array(sorted(class_to_ind, key=lambda k: class_to_ind[k]))
    ind_to_predicates = np.array(sorted(predicate_to_ind,
                                        key=lambda k: predicate_to_ind[k]))

    object_count = {k: v for k, v in sorted(info['object_count'].items(),
                                            key=lambda item: item[1], reverse=True)}
    relation_count = {k: v for k, v in sorted(info['predicate_count'].items(),
                                              key=lambda item: item[1], reverse=True)}

    print('Total number of classes: ', ind_to_classes.shape[0])
    print('Total number of predicates: ', ind_to_predicates.shape[0])

    return ind_to_classes, ind_to_predicates, object_count, relation_count


def get_X_A_F(graph):
    total_objects = list(graph['objects'])
    obj_pair_id = graph['obj_pair_id']
    triplet_to_object_map = graph['triplet_to_object_map']

    N = len(total_objects)
    A = np.zeros((N, N))
    F = np.zeros((N, N))
    for obj_pair, triplet in zip(obj_pair_id, graph['triplets']):
        A[triplet_to_object_map[obj_pair[0]],
          triplet_to_object_map[obj_pair[1]]] = 1
        F[triplet_to_object_map[obj_pair[0]],
          triplet_to_object_map[obj_pair[1]]] = triplet[2] + 1

    X = graph['objects'] + 1

    return X, A, F
