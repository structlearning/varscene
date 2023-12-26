import random
import numpy as np
import torch

FREQUENT_RELATIONS = ['has', 'wearing', 'on', 'behind']

PARENT_OBJ = ['airplane', 'animal', 'bear', 'bike', 'bird', 'boy', 'building', 'car', 'cat', 'child', 'cow', 'dog',
              'elephant', 'giraffe', 'guy', 'girl' 'horse', 'kid', 'lady', 'man', 'men', 'motorcycle', 'people',
              'person', 'plane', 'plant', 'player', 'sheep', 'skier', 'train', 'tree', 'truck', 'vehicle', 'woman',
              'zebra']
CHILD_OBJ = ['arm', 'bag', 'boot', 'branch', 'cap', 'coat', 'cup', 'ear', 'eye', 'face', 'finger', 'fork', 'glove',
             'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'jacket', 'jean', 'leaf', 'leg', 'mouth', 'neck',
             'nose', 'pant', 'paw', 'shirt', 'shoe', 'short', 'skateboard', 'ski', 'sneaker', 'sock', 'surfboard',
             'tail', 'tie', 'tire', 'trunk', 'wheel', 'window', 'windshield', 'wing']

OBJ_DUP_CATEGORY = [
    ('woman', 'guy', 'person', 'man', 'person', 'player', 'skier', 'lady', 'people', 'men'),
    ('zebra', 'animal', 'bear', 'bird', 'cat', 'elephant', 'cow', 'giraffe', 'dog', 'sheep', 'horse'),
    ('boy', 'girl', 'child', 'kid'),
    ('truck', 'vehicle', 'train', 'motorcycle', 'bus', 'boat', 'bike', 'car'),
    ('window', 'windshield'),
    ('building', 'tower', 'house'),
    ('shoe', 'sneaker', 'boot'),
    ('bench', 'chair', 'seat'),
    ('tree', 'trunk'),
    ('food', 'pizza'),
    ('face', 'head', 'mouth'),
    ('fruit', 'banana', 'orange'),
    ('table', 'desk', 'counter', 'stand'),
    ('cabinet', 'shelf', 'drawer'),
    ('coat', 'jacket'),
    ('street', 'track', 'sidewalk'),
    ('airplane', 'plane'),
    ('cap', 'hat', 'helmet'),
    ('hill', 'mountain'),
    ('pole', 'post'),
    ('pot', 'vase'),
    ('jean', 'pant', 'short'),
    ('letter', 'logo', 'sign', 'number'),
    ('laptop', 'screen'),
    ('light', 'lamp'),
    ('beach', 'wave'),
    ('tire', 'wheel'),
    ('paw', 'hand'),
    # following categories have no duplicates
    #  ('shirt'), ('arm'), ('bag'), ('basket'), ('bed'), ('board'), ('book'), ('bottle'), ('bowl'),
    #  ('box'), ('branch'), ('clock'), ('cup'), ('curtain'), ('door'), ('ear'), ('engine'), ('eye'), ('fence'),
    #  ('finger'), ('flag'), ('flower'), ('fork'), ('glass'), ('glove'), ('hair'), ('handle'), ('kite'),
    #  , ('leaf'), ('leg'), ('neck'), ('nose'), ('paper'), ('phone'), ('pillow'), ('plant'),
    #  ('plate'), ('racket'), ('railing'), ('rock'), ('roof'), ('room'), ('sink'), ('skateboard'), ('ski'), ('snow'),
    #  ('sock'), ('surfboard'), ('tail'), ('tie'), ('tile'), ('toilet'), ('towel'), ('umbrella'),
    #  ('vegetable'), ('wing'), ('wire')
]


class VgCleaner:
    def __init__(self, ind_to_classes, ind_to_predicates):
        self.ind_to_classes = ind_to_classes
        self.ind_to_predicates = ind_to_predicates

    def __merge_relations(self, gt_rels, pred_rels):
        relations = dict()
        for gt_rel in gt_rels:
            relations[(gt_rel[0], gt_rel[1])] = gt_rel[2]
        for pred_rel in pred_rels:
            if (pred_rel[0], pred_rel[1]) not in relations:
                relations[(pred_rel[0], pred_rel[1])] = pred_rel[2]
            elif self.ind_to_predicates[relations[(pred_rel[0], pred_rel[1])] - 1] in FREQUENT_RELATIONS:
                relations[(pred_rel[0], pred_rel[1])] = pred_rel[2]
        return relations

    def __remove_multiple_assignments(self, relations, objects):
        child_parent_dict = dict()
        for pair, reln in list(relations.items()):
            if self.ind_to_classes[objects[pair[0]] - 1] in CHILD_OBJ:
                if self.ind_to_classes[objects[pair[1]] - 1] in PARENT_OBJ:
                    if pair[0] not in child_parent_dict:
                        child_parent_dict[pair[0]] = set()
                    child_parent_dict[pair[0]].add(pair[1])
            if self.ind_to_classes[objects[pair[1]] - 1] in CHILD_OBJ:
                if self.ind_to_classes[objects[pair[0]] - 1] in PARENT_OBJ:
                    if pair[1] not in child_parent_dict:
                        child_parent_dict[pair[1]] = set()
                    child_parent_dict[pair[1]].add(pair[0])

        for child, parent_set in list(child_parent_dict.items()):
            parent_list = list(parent_set)
            if len(parent_list) > 1:
                parent_to_keep = random.choice(parent_list)
                parents_to_remove = [p for p in parent_list if p != parent_to_keep]
                for parent in parents_to_remove:
                    if (child, parent) in relations:
                        del relations[(child, parent)]
                    if (parent, child) in relations:
                        del relations[(parent, child)]

        return relations

    @staticmethod
    def __is_same_category(obj1, obj2):
        member = None
        for categ in OBJ_DUP_CATEGORY:
            if obj1 in categ and obj2 in categ:
                member = categ
        if member is not None or obj1 == obj2:
            return True
        else:
            return False

    @staticmethod
    def __member_id(obj, duplicates):
        member_id = None
        for idx, item in enumerate(duplicates):
            if obj in item:
                member_id = idx
        return member_id

    @staticmethod
    def __boxlist_iou(boxlist1, boxlist2):
        """Compute the intersection over union of two set of boxes.
        The box order must be (xmin, ymin, xmax, ymax).
        Arguments:
          boxlist1: (BoxList) bounding boxes, sized [N,4].
          boxlist2: (BoxList) bounding boxes, sized [M,4].
        Returns:
          (tensor) iou, sized [N,M].
        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        """
        if boxlist1.size != boxlist2.size:
            raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
        boxlist1 = boxlist1.convert("xyxy")
        boxlist2 = boxlist2.convert("xyxy")

        area1 = boxlist1.area()
        area2 = boxlist2.area()

        box1, box2 = boxlist1.bbox, boxlist2.bbox

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        to_remove = 1
        wh = (rb - lt + to_remove).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    def __find_duplicate_objects(self, gt, iou_thres):
        duplicates = list(set())
        obj_dict = {k: v for k, v in enumerate(gt.get_field('labels').tolist())}
        # get duplicates using IOU of bbox
        iou_matrix = VgCleaner.__boxlist_iou(gt, gt).numpy()
        list_idx1, list_idx2 = np.nonzero(iou_matrix)
        for idx1, idx2 in zip(list_idx1, list_idx2):
            if iou_matrix[idx1, idx2] > iou_thres and idx1 != idx2:
                obj1 = obj_dict[idx1]
                obj2 = obj_dict[idx2]
                # check match
                if VgCleaner.__is_same_category(self.ind_to_classes[obj1 - 1], self.ind_to_classes[obj2 - 1]):
                    memid1 = VgCleaner.__member_id(idx1, duplicates)
                    memid2 = VgCleaner.__member_id(idx2, duplicates)
                    # if duplicate pair belongs to two different sets, merge them
                    if memid1 is not None and memid2 is not None and memid1 != memid2:
                        for idx in duplicates[memid2]:
                            duplicates[memid1].add(idx)
                        del duplicates[memid2]
                    # if duplicate pair doesnt belong in any set, make a new set
                    elif memid1 is None and memid2 is None:
                        duplicates.append({idx1, idx2})
                    # one of the duplicate belongs to a set
                    elif memid1 is not None and memid2 is None:
                        duplicates[memid1].add(idx2)
                    elif memid1 is None and memid2 is not None:
                        duplicates[memid2].add(idx1)

        return duplicates

    @staticmethod
    def __filter_relations(relations, dups, idx_to_keep):
        # remove edges between duplicate objects
        for pair, reln in list(relations.items()):
            if pair[0] in dups and pair[1] in dups:
                del relations[pair]
        # collect incoming and outgoing edges
        incoming_relns = dict()
        outgoing_relns = dict()
        for pair, reln in relations.items():
            # incoming edges to duplicate objects
            if pair[1] in dups:
                if pair[0] not in incoming_relns:
                    incoming_relns[pair[0]] = (list(), list())
                incoming_relns[pair[0]][0].append(reln)
                incoming_relns[pair[0]][1].append(pair[1])
            # outgoing edges from duplicate objects
            if pair[0] in dups:
                if pair[1] not in outgoing_relns:
                    outgoing_relns[pair[1]] = (list(), list())
                outgoing_relns[pair[1]][0].append(reln)
                outgoing_relns[pair[1]][1].append(pair[0])
        # sample 1 and remove the rest
        for idx, rel_tup in incoming_relns.items():
            unfreq_rels = [reln for reln in rel_tup[0] if reln not in FREQUENT_RELATIONS]
            if len(unfreq_rels) > 0:
                rel = random.choice(unfreq_rels)
            else:
                rel = random.choice(rel_tup[0])
            relations[idx, idx_to_keep] = rel
            for idx2 in rel_tup[1]:
                if idx2 != idx_to_keep:
                    del relations[(idx, idx2)]
        for idx, rel_tup in outgoing_relns.items():
            unfreq_rels = [reln for reln in rel_tup[0] if reln not in FREQUENT_RELATIONS]
            if len(unfreq_rels) > 0:
                rel = random.choice(unfreq_rels)
            else:
                rel = random.choice(rel_tup[0])
            relations[idx_to_keep, idx] = rel
            for idx2 in rel_tup[1]:
                if idx2 != idx_to_keep:
                    del relations[(idx2, idx)]

        return relations

    def clean_and_enrich_edges(self, groundtruth, prediction, predcls_thres, iou_thres):
        # get  ground truth objects
        objects = groundtruth.get_field('labels').tolist()
        # get ground truth relationships
        gt_relation = groundtruth.get_field('relation_tuple').tolist()
        # get predicted relationships
        pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
        pred_rel_label = prediction.get_field('pred_rel_scores')
        pred_rel_label[:, 0] = 0
        pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
        mask = pred_rel_score > predcls_thres
        pred_rel_label = pred_rel_label[mask]
        pred_relations = list()
        for pair, label in zip(pred_rel_pair, pred_rel_label.tolist()):
            pred_relations.append([pair[0], pair[1], label])
        # merge relationships
        relations = self.__merge_relations(gt_relation, pred_relations)
        # remove multipe child-parent assignments
        relations = self.__remove_multiple_assignments(relations, objects)
        # find duplicates
        duplicates = self.__find_duplicate_objects(groundtruth, iou_thres)
        idx_to_delete = list()
        # filter relations
        for dups in duplicates:
            dups = list(dups)
            idx_to_keep = random.choice(dups)
            idx_to_delete.extend([dup for dup in dups if dup != idx_to_keep])
            relations = self.__filter_relations(relations, dups, idx_to_keep)
        filter_idx = [idx for idx in np.arange(len(objects)) if idx not in idx_to_delete]
        # node vector
        x = np.array(objects)
        n = x.shape[0]
        f = np.zeros((n, n))
        # get edges from relations
        for pair, reln in relations.items():
            f[pair] = reln
        # remove duplicates
        x = x[filter_idx]
        f = f[np.ix_(filter_idx, filter_idx)]

        return x, f
