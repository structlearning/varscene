import os
import pickle

import torch

from scene_graph_benchmark.maskrcnn_benchmark.data.datasets.visual_genome import load_info
from uncond_sggen.vg_data.process import VgCleaner

DATA_PATH = '../scene_graph_benchmark/datasets/vg/VG-SGG-with-attri.h5'
EVAL_CHECKPOINT_TRAIN = 'checkpoints/upload_causal_motif_predcls/inference/VG_stanford_filtered_with_attribute_train'
EVAL_CHECKPOINT_TEST = 'checkpoints/upload_causal_motif_predcls/inference/VG_stanford_filtered_with_attribute_train'


def process_vg_data(
        vg_process: VgCleaner,
        eval_path: str,
        save_path: str,
        predcls_thres=0.75,
        iou_thres=0.5
):
    # load detected results
    detected_result = torch.load(os.path.join(eval_path, 'eval_results.pytorch'))
    sg_all = []
    num_samples = len(detected_result['groundtruths'])
    for idx in range(num_samples):
        ground_truth = detected_result['groundtruths'][idx]
        prediction = detected_result['predictions'][idx]
        scene_graph = vg_process.clean_and_enrich_edges(ground_truth, prediction, predcls_thres, iou_thres)

        if scene_graph[0].shape[0] > 0:
            sg_all.append(scene_graph)

    with open(save_path, 'wb') as f:
        pickle.dump(sg_all, f)


if __name__ == '__main__':
    ind_to_classes, ind_to_predicates, _ = load_info(DATA_PATH)
    vg_process = VgCleaner(ind_to_classes, ind_to_predicates)

    process_vg_data(vg_process, EVAL_CHECKPOINT_TRAIN, '../uncond_sggen/vg_data/custom_vg_dataset/train_dataset.p')

    process_vg_data(vg_process, EVAL_CHECKPOINT_TEST, '../uncond_sggen/vg_data/custom_vg_dataset/test_dataset.p')

    with open('../uncond_sggen/vg_data/custom_vg_dataset/categories.p', 'wb') as file:
        pickle.dump((ind_to_classes, ind_to_predicates), file)
