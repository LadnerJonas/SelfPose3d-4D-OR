# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
import time
from tqdm import tqdm
from prettytable import PrettyTable
import copy
import logging

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger, load_backbone_panoptic
import dataset
import models
from utils.vis import save_batch_heatmaps_multi, save_debug_3d_images_all

from utils.vis import save_debug_images_multi


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--with-ssv', dest='with_ssv', action='store_true')
    parser.add_argument('--vis-attn', dest='vis_attn', action='store_true')
    parser.add_argument(
        '--test-file', help='test_file', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()
    with_ssv = args.with_ssv
    #vis_attn = args.vis_attn
    vis_attn = True
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    final_output_dir = os.path.join("./results_publ2/", cfg_name)
    os.makedirs(final_output_dir, exist_ok=True)
    log_file = '{}_{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'),  'eval_map')
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    #gpus = [int(i) for i in config.GPUS.split(',')]
    gpus = [0]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if os.path.isfile(args.test_file):
        test_model_file = args.test_file
    else:
        test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    logger.info('=> test_model_file {}'.format(test_model_file))
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    print(final_output_dir)
    print(with_ssv)
    print(vis_attn)
    model.eval()
    preds, roots = [], []
    with torch.no_grad():
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            if 'panoptic' in config.DATASET.TEST_DATASET or "fdor" in config.DATASET.TEST_DATASET:
                if with_ssv:
                    if vis_attn:
                        #print(inputs)
                        pred, heatmaps, grid_centers, attns = model(views1=inputs, meta1=meta, inference=True, visualize_attn=True)
                        #print(pred)
                        #print(grid_centers)
                        attn_output_dir = os.path.join(final_output_dir, 'attn_vis')
                        if not os.path.exists(attn_output_dir):
                            os.makedirs(attn_output_dir)
                        for k in range(len(inputs)):
                            view_name = "view_{}".format(k + 1)
                            prefix = "{}_{:08}_{}".format(os.path.join(attn_output_dir, "valid"), i, view_name)
                            save_batch_heatmaps_multi(inputs[k], attns[k], "{}_hm_attn.jpg".format(prefix))

                            save_debug_images_multi(
                                config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix
                            )
                    else:
                        #print(i)
                        #print(inputs)
                        pred, heatmaps, grid_centers = model(views1=inputs, meta1=meta, inference=True)
                        #print(pred)
                        #print(heatmaps)
                        #print(grid_centers)
                else:
                    pred, _, grid_centers, _, _, _ = model(views=inputs, meta=meta)
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pred, _, _, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)

            pred = pred.detach().cpu().numpy()
            root = grid_centers.detach().cpu().numpy()
            targets_3d = targets_3d[0].cpu().numpy()
            #print(f"pred: {pred}")
            #print(f"root: {root}")
            #print(f"targets_3d: {targets_3d}")
            for b in range(pred.shape[0]):
                preds.append(pred[b])
                roots.append(root[b])
        if 'voxelpose' in config.DATASET.TEST_DATASET:
            actor_pcp, avg_pcp, _, recall = test_dataset.evaluate(preds)
            msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
                  ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}'.format(
                pcp_1=actor_pcp[0] * 100, pcp_2=actor_pcp[1] * 100, pcp_3=actor_pcp[2] * 100, pcp_avg=avg_pcp * 100, recall=recall)
            print(msg)

        elif 'panoptic' in config.DATASET.TEST_DATASET or "fdor" in config.DATASET.TEST_DATASET:
            mpjpe_threshold = np.arange(25, 155, 25)
            aps_all, recs_all, mpjpe_all, avg_recall_all = test_dataset.evaluate(preds, roots, final_output_dir)
            types_eval = ["pose", "root"]
            for aps, recs, mpjpe, recall, type_eval in zip(aps_all, recs_all, mpjpe_all, avg_recall_all, types_eval):
                tb = PrettyTable()
                print(f'Type: {type_eval}')
                tb.field_names = ['Threshold/mm'] + [f'{i}' for i in mpjpe_threshold]
                tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
                tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
                print(tb)
                print(f'MPJPE: {mpjpe:.2f}mm')
                print(f'recall@500: {recall:.4f}, {np.array(recs).mean()}')
        else:
            tb = PrettyTable()
            actor_pcp, avg_pcp, bone_person_pcp, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Bone Group'] + [f'Actor {i+1}' for i in range(len(actor_pcp))] + ['Average']
            for k, v in bone_person_pcp.items():
                tb.add_row([k] + [f'{i*100:.1f}' for i in v] + [f'{np.mean(v)*100:.1f}'])
            tb.add_row(['Total'] + [f'{i*100:.1f}' for i in actor_pcp] + [f'{avg_pcp*100:.1f}'])
            print(tb)


if __name__ == "__main__":
    main()
