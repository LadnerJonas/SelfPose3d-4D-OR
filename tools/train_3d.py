'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch import autograd
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json

import _init_paths
from core.config import config
from core.config import update_config
from core.function import train_3d, train_3d_ssv, validate_3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint, load_model_state
from utils.utils import load_backbone_panoptic
import dataset
import models
import random
import numpy as np


#parser.add_argument("--cfg", help="experiment configure file name", required=False, default="./configs/panoptic_ssv/resnet50/run_18_1_train_pseudo_gt_3d.yaml", type=str)

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def get_optimizer(model):
    lr = config.TRAIN.LR
    with_root_net = not config.NETWORK.USE_GT  # USE_GT means using GT proposals for pose regression
    freeze_root_net = config.NETWORK.FREEZE_ROOTNET
    train_backbone = config.NETWORK.TRAIN_BACKBONE
    if train_backbone:
        if model.module.backbone is not None:
            for params in model.module.backbone.parameters():
                params.requires_grad = True
    else:
        if model.module.backbone is not None:
            for params in model.module.backbone.parameters():
                params.requires_grad = False
    if not config.NETWORK.TRAIN_ONLY_2D:
        if not config.NETWORK.TRAIN_ONLY_ROOTNET:
            for params in model.module.pose_net.parameters():
                params.requires_grad = True
        if with_root_net:
            if freeze_root_net:
                for params in model.module.root_net.parameters():
                    params.requires_grad = False
            else:
                for params in model.module.root_net.parameters():
                    params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    # optimizer = optim.Adam(model.module.parameters(), lr=lr)

    return model, optimizer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def main():
    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    args = parse_args()
    with_ssv = config.WITH_SSV  # whether using self-supervised learning
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "train")

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(",")]
    print(f"gpus: {gpus}")
    print("=> Loading data ..")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    logger.info(config.DATASET)
    logger.info(json.dumps(config.DATASET))
    train_dataset = eval("dataset." + config.DATASET.TRAIN_DATASET)(
        config,
        config.DATASET.TRAIN_SUBSET,
        True,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    print(len(train_dataset))
    #for i, x in enumerate(train_dataset):

        #print(f"Batch {i}: data: {x}")
    print("train batch size: ", config.TRAIN.BATCH_SIZE * len(gpus))

    #train_dataset = CustomDataset(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
    )

    print("read out train data loader: len ", len(train_loader))
    #for i, x in enumerate(train_loader):
        #print(f"Batch {i}: Inputs shape: {x}")

    test_dataset = eval("dataset." + config.DATASET.TEST_DATASET)(
        config,
        config.DATASET.TEST_SUBSET,
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    print("test dataset: ", test_dataset, " length:", len(test_dataset))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False,
    )

    print("read out test data loader: len ", len(test_loader))
    #for i, (data, label) in enumerate(test_loader):
    #    print(f"Batch {i}: Data shape: {data.shape}, Label shape: {label.shape}")

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print("=> Constructing models ..")
    model = eval("models." + config.MODEL + ".get_multi_person_pose_net")(config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    #print("=> model ->", model)

    model, optimizer = get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    last_epoch = -1

    best_precision = 0
    if config.NETWORK.PRETRAINED_BACKBONE:
        if config.NETWORK.PRETRAINED_BACKBONE_PSEUDOGT:
            print("=> loading backbone from =", config.NETWORK.PRETRAINED_BACKBONE)
            st_dict = {
                k.replace("backbone.", ""): v
                for k, v in torch.load(config.NETWORK.PRETRAINED_BACKBONE).items()
                if "backbone" in k
            }
            mk, uk = model.module.backbone.load_state_dict(st_dict, strict=True)
            print("=> missing keys in backbone =", mk)
            print("=> unexpected keys in backbone =", uk)
        else:
            model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)

    if config.NETWORK.INIT_ROOTNET:
        print("=> loading rootnet from =", config.NETWORK.INIT_ROOTNET)
        st_dict = {
            k.replace("root_net.", ""): v
            for k, v in torch.load(config.NETWORK.INIT_ROOTNET).items()
            if "root_net" in k
        }
        mk, uk = model.module.root_net.load_state_dict(st_dict, strict=True)
        print("=> missing keys in rootnet =", mk)
        print("=> unexpected keys in rootnet=", uk)

    if config.NETWORK.INIT_ALL:
        print("=> loading all from =", config.NETWORK.INIT_ALL)
        st_dict = torch.load(config.NETWORK.INIT_ALL)
        mk, uk = model.module.load_state_dict(st_dict, strict=True)
        print("=> missing keys in all =", mk)
        print("=> unexpected keys in all =", uk)

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision, last_epoch = load_checkpoint(
            model, optimizer, final_output_dir
        )

    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    print("=> Training...")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )
    gpu_memory_usage = torch.cuda.memory_allocated(0) / 1024.0 / 1024.0 / 1024.0
    print("GPU memory usage: {:.2f} GB (init before cache clear)".format(gpu_memory_usage))
    torch.cuda.empty_cache()
    gpu_memory_usage = torch.cuda.memory_allocated(0) / 1024.0 / 1024.0 / 1024.0
    print("GPU memory usage: {:.2f} GB (init after cache clear".format(gpu_memory_usage))

    #torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, end_epoch):
        print("Epoch: {}".format(epoch))
        logger.info("learning rate for this epoch {}".format(lr_scheduler.get_last_lr()))
        if with_ssv:
            train_3d_ssv(
                config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict
            )
        else:
            train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        lr_scheduler.step()
        if config.NETWORK.TRAIN_ONLY_2D:
            precision = None
        else:
            precision = validate_3d(
                config, model, test_loader, epoch, final_output_dir, with_ssv=with_ssv
            )

        if precision is not None and precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False

        logger.info("=> saving checkpoint to {} (Best: {})".format(final_output_dir, best_model))
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),
                "precision": best_precision,
                "optimizer": optimizer.state_dict(),
            },
            best_model,
            final_output_dir,
        )

    final_model_state_file = os.path.join(final_output_dir, "final_state.pth.tar")
    logger.info("saving final model state to {}".format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict["writer"].close()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
