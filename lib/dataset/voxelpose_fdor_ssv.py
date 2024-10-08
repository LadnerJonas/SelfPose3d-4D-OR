# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

import copy
import json_tricks as json
import numpy as np
import os
import scipy.io as scio
from scipy.spatial.transform.rotation import Rotation

from dataset.JointsDatasetSSV import JointsDatasetSSV
from utils.transforms import projectPoints
import open3d as o3d

OR_4D_JOINTS_DEF = {
    'head': 0,
    'neck': 1,
    'left_shoulder': 2,
    'right_shoulder': 3,
    'left_hip': 4,
    'right_hip': 5,
    'left_elbow': 6,
    'right_elbow': 7,
    'left_wrist': 8,
    'right_wrist': 9,
    'left_knee': 10,
    'right_knee': 11,
    'leftfoot': 12,
    'rightfoot': 13
}

LIMBS = [
    [5, 4],  # (righthip-lefthip)
    [9, 7],  # (rightwrist - rightelbow)
    [7, 3],  # (rightelbow - rightshoulder)
    [2, 6],  # (leftshoulder - leftelbow)
    [6, 8],  # (leftelbow - leftwrist)
    [5, 3],  # (righthip - rightshoulder)
    [4, 2],  # (lefthip - leftshoulder)
    [3, 1],  # (rightshoulder - neck)
    [2, 1],  # (leftshoulder - neck)
    [1, 0],  # (neck - head)
    [10, 4],  # (leftknee,lefthip),
    [11, 5],  # (rightknee,righthip),
    [12, 10],  # (leftfoot,leftknee),
    [13, 11]  # (rightfoot,rightknee),

]

IDX_TO_BODY_PART = ['head', 'neck', 'leftshoulder', 'rightshoulder', 'lefthip', 'righthip', 'leftelbow', 'rightelbow', 'leftwrist', 'rightwrist', 'leftknee',
                    'rightknee', 'leftfoot', 'rightfoot']


def coord_transform_human_pose_tool_to_OR_4D(arr):
    # reverse of coord_transform_OR_4D_to_human_pose_tool
    arr *= 25
    arr[:, 2] += 1000
    arr[:, 1] *= -1
    arr = arr[:, [0, 2, 1]]
    return arr


TAKE_SPLIT = {'train': [1, 3, 5, 7, 9, 10], 'validation': [4, 8], 'test': [2, 6]}
#TAKE_SPLIT = {'train': [1, 2, 3, 4, 5, 7, 9, 10], 'validation': [8], 'test': [6]}
#TAKE_SPLIT = {'train': [5], 'validation': [4], 'test': [2]}


class Voxelpose_fdor_ssv(JointsDatasetSSV):
    def __init__(self, cfg, image_set, is_train, transform=None, inference=False):
        super().__init__(cfg, image_set, is_train, [], transform)
        self.pixel_std = 200.0
        self.joints_def = OR_4D_JOINTS_DEF
        print(f'transform is none: {transform is None}')
        self.take_indices = TAKE_SPLIT[image_set]
        self.camera_num_total = cfg.DATASET.CAMERA_NUM_TOTAL
        print(f'{image_set} using indices: {self.take_indices}')

        self.take_to_annotations = {}
        self.take_to_timestamp_to_pcd_and_frames_list = {}

        for take_idx in self.take_indices:
            annotations_path = Path(f'/home/data/4D-OR/export_holistic_take{take_idx}_processed/2D_keypoint_annotations.json')
            with annotations_path.open() as f:
                annotations = json.load(f)
                self.take_to_annotations[take_idx] = annotations

            with open(f'/home/data/4D-OR/export_holistic_take{take_idx}_processed/timestamp_to_pcd_and_frames_list.json') as f:
                timestamp_to_pcd_and_frames_list = json.load(f)
                self.take_to_timestamp_to_pcd_and_frames_list[take_idx] = timestamp_to_pcd_and_frames_list

        self.limbs = LIMBS
        self.num_joints = len(OR_4D_JOINTS_DEF)
        self.cam_list = [1, 2, 3, 4, 5]
        self.num_views = len(self.cam_list)

        self.pred_pose2d = self._get_pred_pose2d()
        self.inference = inference
        self.db = self._get_db()

        self.db_size = len(self.db)
        print(f'Using {image_set} imageset')

    def _get_pred_pose2d(self):
        pred_2d = np.load(f'/home/guests/jonas_ladner/SelfPose3dParentFolder/SelfPose3d/data/HigherHRNet_files/pred_or_4d_hrnet_coco_{self.image_set}.npz', allow_pickle=True)['arr_0'].item()
        return pred_2d

    def get_image_dicts(self):
        image_dicts = []
        image_id_counter = 0

        for take_idx in self.take_indices:
            for idx, (_, corresponding_channels) in enumerate(self.take_to_timestamp_to_pcd_and_frames_list[take_idx]):
                for c_idx in range(1, 7):
                    rgb_str = corresponding_channels[f'color_{c_idx}']
                    image_name = f'camera0{c_idx}_colorimage-{rgb_str}.jpg'
                    image_path = f'/home/data/4D-OR/export_holistic_take{take_idx}_processed/colorimage/{image_name}'
                    image_dict = {'take_idx': take_idx, 'cam': c_idx, 'image_name': image_name, 'image_id': image_id_counter, 'image_path': image_path}
                    if len([elem for elem in image_dicts if elem['image_path'] == image_path]) > 0:  # Make sure the image is not already included
                        continue
                    image_id_counter += 1
                    image_dicts.append(image_dict)

        return image_dicts

    def transformGivenTrfMatrix(self, ann3d, tr_mat):
        """
        Transform the 3D points from one coordiate system to another given 4x4 transformation matrix
        :param ann3d: input 3D points
        :param tr_mat: 4x4 transformation matrix
        :return: transformed 3D points
        """
        X = ann3d[:, 0]
        Y = ann3d[:, 1]
        Z = ann3d[:, 2]
        pt3d = np.vstack((X, Y, Z, np.ones(len(X))))
        # transform points to room coordinate
        pt3d = np.dot(tr_mat, pt3d)
        pt3d = pt3d[0:3]
        return pt3d

    def projectCam3DTo2D(self, pose3D, K):
        """
        Project 3D point in camera coordinates to image given intrinsic camera parameters (focal-length and principal-point)
        :param pose3D: 3D points in camera coordinates
        :param camparam: intrinsic camera parameters
        :return: 2D point on the images
        """
        focal = K[0, 0], K[1, 1]
        pp = K[0, 2], K[1, 2]
        pose3D[2][pose3D[2] == 0.0] = 1.0  # replace zero with 1 to avoid divide by zeros
        p1 = ((np.divide(pose3D[0], pose3D[2])) * focal[0]) + pp[0]
        p2 = ((np.divide(pose3D[1], pose3D[2])) * focal[1]) + pp[1]
        return np.vstack((p1, p2))

    def _get_db(self):
        width = 2048
        height = 1536

        db = []
        cameras = self._get_cam(root_path=Path(f'/home/data/4D-OR/export_holistic_take{self.take_indices[0]}_processed'))
        image_dicts = self.get_image_dicts()
        tmp_dict = {}
        for image_dict in image_dicts:
            tmp_dict[f'{image_dict["take_idx"]}_{image_dict["image_name"]}'] = image_dict

        image_dicts = tmp_dict

        for take_idx in self.take_indices:
            for idx, (_, corresponding_channels) in enumerate(self.take_to_timestamp_to_pcd_and_frames_list[take_idx]):
                pcd_idx_str = corresponding_channels['pcd']
                human_pose_json_path = Path(f'/home/data/4D-OR/export_holistic_take{take_idx}_processed/annotations') / f'{pcd_idx_str}.json'
                bodies = []
                is_patient_mask = []
                if human_pose_json_path.exists():
                    with human_pose_json_path.open() as f:
                        human_pose_json = json.load(f)
                        human_names = {elem['humanName'] for elem in human_pose_json['labels']}
                        for _, human_name in enumerate(human_names):
                            human_pose = []
                            human_joints = [elem for elem in human_pose_json['labels'] if elem['humanName'] == human_name]
                            joint_positions = {}
                            for human_joint in human_joints:
                                joint_positions[human_joint['jointName']] = (
                                    human_joint['point3d']['location']['x'], human_joint['point3d']['location']['y'], human_joint['point3d']['location']['z'])

                            for body_part in IDX_TO_BODY_PART:
                                human_pose.append(joint_positions[body_part])
                            human_pose = np.asarray(human_pose)
                            human_pose = coord_transform_human_pose_tool_to_OR_4D(human_pose)
                            # human_pose /= 500
                            human_pose = np.stack(
                                [human_pose[:, 0], human_pose[:, 1], human_pose[:, 2], np.ones(len(human_pose)) + 1]).transpose().flatten().tolist()
                            bodies.append(human_pose)
                            is_patient_mask.append(human_name == 'Patient')
                if len(bodies) == 0 and not self.inference:
                    continue
                for k, cam in cameras.items():
                    color_image_idx_str = corresponding_channels[f'color_{k}']
                    identifier = f'{take_idx}_camera0{k}_colorimage-{color_image_idx_str}.jpg'
                    image_dict = image_dicts[identifier]
                    preds = self.pred_pose2d[identifier]
                    preds = [np.array(p) for p in
                             preds]

                    all_poses_3d = []
                    all_is_patient = []
                    all_poses_vis_3d = []
                    all_poses = []
                    all_poses_vis = []
                    for is_patient, body in zip(is_patient_mask, bodies):
                        pose3d = np.array(body).reshape((-1, 4))
                        pose3d = pose3d[:self.num_joints]

                        joints_vis = pose3d[:, -1] > 0.1

                        if not joints_vis[self.root_id[0]] or not joints_vis[self.root_id[1]]:
                            continue

                        all_poses_3d.append(pose3d[:, 0:3])
                        all_is_patient.append(is_patient)
                        all_poses_vis_3d.append(
                            np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                        pose3d_pc = o3d.geometry.PointCloud()
                        pose3d_pc.points = o3d.utility.Vector3dVector(pose3d[:, :3] / 500)
                        pose3d_pc.transform(np.linalg.inv(cam['extrinsics']))  # Bring from world to rgb camera coords
                        pose3d_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # this is needed
                        obj_points = np.asarray(pose3d_pc.points)
                        # Project onto image
                        obj_points[:, 2][obj_points[:, 2] == 0.0] = 1.0  # replace zero with 1 to avoid divide by zeros
                        x = obj_points[:, 0]
                        y = obj_points[:, 1]
                        z = obj_points[:, 2]

                        fx = cam['fx']
                        fy = cam['fy']
                        cx = cam['cx']
                        cy = cam['cy']
                        # Project points without distortion
                        u = (x * fx / z) + cx
                        v = (y * fy / z) + cy

                        # Distortion coefficients
                        k1, k2, p1, p2, k3 = cam['distCoef']

                        # Scale u and v back to the original range before distortion
                        u_normalized = (u - cx) / fx
                        v_normalized = (v - cy) / fy

                        # Calculate radial distortion
                        r2 = u_normalized**2 + v_normalized**2
                        radial_distortion = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

                        # Calculate tangential distortion
                        tan_distortion_x = 2 * p1 * u_normalized * v_normalized + p2 * (r2 + 2 * u_normalized**2)
                        tan_distortion_y = p1 * (r2 + 2 * v_normalized**2) + 2 * p2 * u_normalized * v_normalized


                        # Apply distortion corrections
                        u_distorted_normalized = u_normalized * radial_distortion + tan_distortion_x
                        v_distorted_normalized = v_normalized * radial_distortion + tan_distortion_y

                        # Scale back to pixel coordinates
                        u_distorted = u_distorted_normalized * fx + cx
                        v_distorted = v_distorted_normalized * fy + cy

                        # Final 2D pose points
                        pose2d = np.stack([u_distorted, v_distorted], axis=1)
                        # pose2d = np.stack([u, v], axis=1)
                        x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                 pose2d[:, 0] <= width - 1)
                        y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                 pose2d[:, 1] <= height - 1)
                        check = np.bitwise_and(x_check, y_check)
                        joints_vis[np.logical_not(check)] = 0

                        all_poses.append(pose2d)
                        all_poses_vis.append(
                            np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                    if len(all_poses_3d) > 0 or self.inference:
                        our_cam = {}
                        our_cam['R'] = cam['R']
                        our_cam['T'] = cam['T']
                        our_cam['fx'] = cam['fx']
                        our_cam['fy'] = cam['fy']
                        our_cam['cx'] = cam['cx']
                        our_cam['cy'] = cam['cy']
                        our_cam['k'] = cam['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = cam['distCoef'][[2, 3]].reshape(2, 1)
                        our_cam['extrinsics'] = cam['extrinsics']

                        db.append({
                            'key': f"{identifier}",
                            'image': image_dict['image_path'],
                            'joints_3d': all_poses_3d,
                            #'is_patient_mask': all_is_patient,
                            'joints_3d_vis': all_poses_vis_3d,
                            'joints_2d': all_poses,
                            'joints_2d_vis': all_poses_vis,
                            'camera': our_cam,
                            'pred_pose2d': preds,
                            'pcd_idx_str': pcd_idx_str,
                            'take_idx': take_idx
                        })
        print(f'Loaded {len(db)} samples in voxelpose_fdor_ssv')
        return db

    def _get_cam(self, root_path, cam_count=5):
        cameras = {}
        for c_idx in range(1, cam_count + 1):
            cam_json_path = root_path / f'camera0{c_idx}.json'
            with cam_json_path.open() as f:
                cam_info = json.load(f)['value0']
                intrinsics_json = cam_info['color_parameters']['intrinsics_matrix']
                intrinsics = np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                                         [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                                         [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])

                extrinsics_json = cam_info['camera_pose']
                trans = extrinsics_json['translation']
                rot = extrinsics_json['rotation']
                extrinsics = np.zeros((4, 4), dtype=np.float32)
                rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
                extrinsics[:3, :3] = rot_matrix
                extrinsics[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]

                color2depth_json = cam_info['color2depth_transform']
                trans = color2depth_json['translation']
                rot = color2depth_json['rotation']
                color2depth_transform = np.zeros((4, 4), dtype=np.float32)
                rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
                color2depth_transform[:3, :3] = rot_matrix
                color2depth_transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
                extrinsics = np.matmul(extrinsics, color2depth_transform)  # Extrinsics were given for the depth camera, convert them to color camera

                fov_x = cam_info['color_parameters']['fov_x']
                fov_y = cam_info['color_parameters']['fov_y']
                c_x = cam_info['color_parameters']['c_x']
                c_y = cam_info['color_parameters']['c_y']

                # Extract radial and tangential distortion coefficients
                radial_distortion_json = cam_info['color_parameters']['radial_distortion']
                tangential_distortion_json = cam_info['color_parameters']['tangential_distortion']

                # Radial distortion: Assume the first 3 radial coefficients are relevant
                k1 = radial_distortion_json['m00']
                k2 = radial_distortion_json['m10']
                k3 = radial_distortion_json['m20']

                # Tangential distortion: p1 and p2
                p1 = tangential_distortion_json['m00']
                p2 = tangential_distortion_json['m10']

                # Create the distortion coefficients array with 5 elements: k1, k2, p1, p2, k3
                distCoef = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

                cameras[str(c_idx)] = {'K': intrinsics, 'distCoef': distCoef, 'R': extrinsics[:3, :3], 'T': np.expand_dims(extrinsics[:3, 3], axis=1),
                                       'fx': np.asarray(fov_x), 'fy': np.asarray(fov_y), 'cx': np.asarray(c_x), 'cy': np.asarray(c_y), 'extrinsics': extrinsics}
        return cameras

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return self.db_size // self.num_views

def evaluate(self, preds):
    eval_list = []
    gt_num = self.db_size // self.num_views
    assert len(preds) == gt_num, "number mismatch"

    total_gt = 0
    for i in range(gt_num):
        index = self.num_views * i
        db_rec = copy.deepcopy(self.db[index])
        joints_3d = db_rec["joints_3d"]
        joints_3d_vis = db_rec["joints_3d_vis"]

        if len(joints_3d) == 0:
            continue

        pred = preds[i].copy()
        pred = pred[pred[:, 0, 3] >= 0]
        for pose in pred:
            mpjpes = []
            for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                vis = gt_vis[:, 0] > 0
                #print("gt: ", gt[vis], "pose: ", pose[vis, 0:3])
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

        total_gt += len(joints_3d)

    mpjpe_threshold = np.arange(25, 155, 25)
    aps = []
    recs = []
    for t in mpjpe_threshold:
        ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
        aps.append(ap)
        recs.append(rec)

    return (
        aps,
        recs,
        self._eval_list_to_mpjpe(eval_list),
        self._eval_list_to_recall(eval_list, total_gt),
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
