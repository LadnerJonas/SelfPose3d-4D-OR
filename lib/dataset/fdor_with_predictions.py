'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy

from dataset.JointsDataset import JointsDataset
from utils.transforms import fliplr_joints, projectPoints

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    #"holistic_take1",
    #"holistic_take2",
    "holistic_take3",
    "holistic_take4",
    "holistic_take5",
    #"holistic_take6",
    #"holistic_take7",
    #"holistic_take8",
    #"holistic_take9",
    #"holistic_take10",
]
VAL_LIST = [
    #"holistic_take1",
    "holistic_take2",
    #"holistic_take3",
    #"holistic_take4",
    #"holistic_take5",
]

JOINTS_DEF = {
    "neck": 0,
    "nose": 1,
    "mid-hip": 2,
    "l-shoulder": 3,
    "l-elbow": 4,
    "l-wrist": 5,
    "l-hip": 6,
    "l-knee": 7,
    "l-ankle": 8,
    "r-shoulder": 9,
    "r-elbow": 10,
    "r-wrist": 11,
    "r-hip": 12,
    "r-knee": 13,
    "r-ankle": 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}
FLIP_LR_JOINTS15 = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]

LIMBS = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 9],
    [9, 10],
    [10, 11],
    [2, 6],
    [2, 12],
    [6, 7],
    [7, 8],
    [12, 13],
    [13, 14],
]


class Fdor_with_predictions(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        print("inside fdor-with-predictions.py")
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)
        ROOT = "./data"
        self.dataset_suffix = cfg.DATASET.SUFFIX if is_train else "sub"
        #self.camera_num_total = cfg.DATASET.CAMERA_NUM_TOTAL
        self.camera_num_total = 6
        #self.cameras = cfg.DATASET.CAMERAS
        self.cameras = [0,1,2,3,4,5] # [0,2,4]
        print(self.cameras)
        # Camera_num_total is set to 5. 
        # Cameras is a list referring to the camera index. e.g. [0,1,2,3,4] / [0,2,3]
        # We always read the same pickle, and then select the relevant data based on camera index

        if self.image_set == "train":
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            cam_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
            self.cam_list = []
            for idx in self.cameras:
                self.cam_list.append(cam_list[idx]) # select the camera based on camera index
            print(self.cam_list)
        elif self.image_set == "validation":
            self.sequence_list = VAL_LIST
            self._interval = 12
            cam_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
            self.cam_list = []
            for idx in self.cameras:
                self.cam_list.append(cam_list[idx]) # select the camera based on camera index

        self.db_file = "group_{}_cam{}_{}_with_predictions.pkl".format(
            self.image_set, self.camera_num_total, self.dataset_suffix
        )
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            print("=> loading the pickle file = ", self.db_file)
            info = pickle.load(open(self.db_file, "rb"))
            print(self.sequence_list)
            # assert info["sequence_list"] == self.sequence_list
            assert info["interval"] == self._interval
            # assert info["cam_list"] == self.cam_list
            print(ROOT)
            self.db = info["db"]
            for p in info["db"]:
                p["image"] = os.path.join(ROOT, p["image"])
            print("=> self.db", len(self.db))
        else:
            self.db = self._get_db()
            info = {
                "sequence_list": self.sequence_list,
                "interval": self._interval,
                "cam_list": self.cam_list,
                "db": self.db,
            }
            pickle.dump(info, open(self.db_file, "wb"))
        # self.db = self._get_db()
        self.db_size = len(self.db)
        print(f"db_size {self.db_size}")

    def _get_db(self):
        width = 2048
        height = 1536
        #width = 1440
        #height = 1080
        db = []
        for seq in self.sequence_list:

            cameras = self._get_cam(seq)

            curr_anno = osp.join(
                self.dataset_root, seq, "hdExtractedPredictions"
            )
            anno_files = sorted(glob.iglob("{:s}/*.json".format(curr_anno)))
            print("=> loading annotations from {:s}".format(curr_anno))
            print(anno_files)

            for i, file in enumerate(anno_files):
                #if i % self._interval == 0 or : # not needed as annotation files are already sparse
                if True:
                    with open(file) as dfile:
                        bodies = json.load(dfile)["bodies"]
                    if len(bodies) == 0:
                        continue

                    for k, v in cameras.items():
                        postfix = osp.basename(file).replace("prediction-", "")
                        image = osp.join(
                            seq, "hdImgs", "camera{:02d}".format(k[1]) + "_colorimage-" + postfix
                        #    seq, "hdImgs-1440-1080", "camera{:02d}".format(k[1]) + "_colorimage-" + postfix
                        )
                        image = image.replace("json", "jpg")
                        if not osp.exists(osp.join(self.dataset_root, image)):
                            continue
                        print(f"image for {k} and {v}")
                        print(image)

                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        for body in bodies:
                            pose3d = np.array(body["joints19"]).reshape((-1, 4))
                            pose3d = pose3d[: self.num_joints]

                            joints_vis = pose3d[:, -1] > 0.1

                            if not joints_vis[self.root_id]:
                                continue

                            # Coordinate transformation
                            M = np.array(
                                [
                                    [1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0],
                                    [0.0, 1.0, 0.0],
                                ]
                            )
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                            all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1
                                )
                            )

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(),
                                v["K"],
                                v["R"],
                                v["t"],
                                v["distCoef"],
                            ).transpose()[:, :2]
                            x_check = np.bitwise_and(
                                pose2d[:, 0] >= 0, pose2d[:, 0] <= width - 1
                            )
                            y_check = np.bitwise_and(
                                pose2d[:, 1] >= 0, pose2d[:, 1] <= height - 1
                            )
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0

                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1
                                )
                            )

                        if len(all_poses_3d) > 0:
                            our_cam = {}
                            our_cam["R"] = v["R"]
                            our_cam["T"] = (
                                -np.dot(v["R"].T, v["t"]) * 10.0
                            )  # cm to mm
                            our_cam["fx"] = np.array(v["K"][0, 0])
                            our_cam["fy"] = np.array(v["K"][1, 1])
                            our_cam["cx"] = np.array(v["K"][0, 2])
                            our_cam["cy"] = np.array(v["K"][1, 2])
                            our_cam["k"] = v["distCoef"][[0, 1, 4]].reshape(
                                3, 1
                            )
                            our_cam["p"] = v["distCoef"][[2, 3]].reshape(2, 1)

                            db.append(
                                {
                                    "key": "{}_{}{}".format(
                                        seq, "{:02d}".format(k[1]), postfix.split(".")[0]
                                    ),
                                    "image": osp.join(self.dataset_root, image),
                                    "joints_3d": all_poses_3d,
                                    "joints_3d_vis": all_poses_vis_3d,
                                    "joints_2d": all_poses,
                                    "joints_2d_vis": all_poses_vis,
                                    "camera": our_cam,
                                }
                            )
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(
            self.dataset_root, seq, "calibration_{:s}.json".format(seq)
        )
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib["cameras"]:
            if (cam["panel"], cam["node"]) in self.cam_list:
                sel_cam = {}
                sel_cam["K"] = np.array(cam["K"])
                sel_cam["distCoef"] = np.array(cam["distCoef"])
                sel_cam["R"] = np.array(cam["R"]).dot(M)
                sel_cam["t"] = np.array(cam["t"]).reshape((3, 1))
                cameras[(cam["panel"], cam["node"])] = sel_cam
        return cameras

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for k in range(self.camera_num_total):
            i, t, w, t3, m, ih = super().__getitem__(self.camera_num_total * idx + self.cameras[k])
            if i is None:
                print(f"error: {self.camera_num_total * idx + self.cameras[k]}, {self.camera_num_total} {idx} {self.cameras[k]} {k}")
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.camera_num_total

    def evaluate(self, preds, roots=None, output_dir=""):
        eval_list, eval_list_root = [], []
        gt_num = self.db_size // self.camera_num_total
        assert len(preds) == gt_num, "number mismatch"

        total_gt = 0
        for i in range(gt_num):
            index = self.camera_num_total * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec["joints_3d"]
            joints_3d_vis = db_rec["joints_3d_vis"]
            joints_3d_root = [a[self.root_id] for a in db_rec["joints_3d"]]
            joints_3d_vis_root = [
                a[self.root_id] for a in db_rec["joints_3d_vis"]
            ]

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]

            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)
                        )
                    )
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append(
                    {
                        "mpjpe": float(min_mpjpe),
                        "score": float(score),
                        "gt_id": int(total_gt + min_gt),
                    }
                )

            root = roots[i].copy()
            root = root[root[:, 3] >= 0]
            for rt in root:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d_root, joints_3d_vis_root):
                    vis = gt_vis[0] > 0
                    if vis:
                        mpjpe = np.mean(
                            np.sqrt(np.sum((rt[0:3] - gt) ** 2, axis=-1))
                        )
                        mpjpes.append(mpjpe)
                if len(mpjpes) > 0:
                    min_gt = np.argmin(mpjpes)
                    min_mpjpe = np.min(mpjpes)
                    score = rt[4]
                    eval_list_root.append(
                        {
                            "mpjpe": float(min_mpjpe),
                            "score": float(score),
                            "gt_id": int(total_gt + min_gt),
                        }
                    )
            self.db[index]["preds_3d"] = pred
            self.db[index]["roots_3d"] = root
            total_gt += len(joints_3d)

        if output_dir:
            output_file = os.path.join(output_dir, "predictions_dump.pkl")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            print("dumping the results at :", output_file)
            pickle.dump(self.db, open(output_file, "wb"))

        mpjpe_threshold = np.arange(25, 155, 25)
        aps, aps_root = [], []
        recs, recs_root = [], []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            ap_root, rec_root = self._eval_list_to_ap(
                eval_list_root, total_gt, t
            )
            aps.append(ap)
            recs.append(rec)
            aps_root.append(ap_root)
            recs_root.append(rec_root)
        mpjpe_res = self._eval_list_to_mpjpe(eval_list)
        mpjpe_res_root = self._eval_list_to_mpjpe(eval_list_root)
        recall_res = self._eval_list_to_recall(eval_list, total_gt)
        recall_res_root = self._eval_list_to_recall(eval_list_root, total_gt)

        return (
            (aps, aps_root),
            (recs, recs_root),
            (mpjpe_res, mpjpe_res_root),
            (recall_res, recall_res_root),
        )

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt
