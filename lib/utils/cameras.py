'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


from __future__ import division
import torch
import numpy as np


def unfold_camera_param(camera, device=None):
    R = torch.as_tensor(camera["R"], dtype=torch.float, device=device)
    T = torch.as_tensor(camera["T"], dtype=torch.float, device=device)
    fx = torch.as_tensor(camera["fx"], dtype=torch.float, device=device)
    fy = torch.as_tensor(camera["fy"], dtype=torch.float, device=device)
    f = torch.tensor([fx, fy], dtype=torch.float, device=device).reshape(2, 1)
    c = torch.as_tensor(
        [[camera["cx"]], [camera["cy"]]], dtype=torch.float, device=device
    )
    k = torch.as_tensor(camera["k"], dtype=torch.float, device=device)
    p = torch.as_tensor(camera["p"], dtype=torch.float, device=device)
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    xcam = torch.mm(R, torch.t(x) - T)
    y = xcam[:2] / (xcam[2] + 1e-5)

    kexp = k.repeat((1, n))
    r2 = torch.sum(y ** 2, 0, keepdim=True)
    r2 = torch.clamp(r2, max=1e10)
    r2exp = torch.cat([r2, r2 ** 2, r2 ** 3], 0)
    radial = 1 + torch.einsum("ij,ij->j", kexp, r2exp)

    tan = p[0] * y[1] + p[1] * y[0]
    corr = (radial + 2 * tan).repeat((2, 1))

    y = y * corr + torch.ger(torch.cat([p[1], p[0]]).view(-1), r2.view(-1))
    ypixel = (f * y) + c
    return torch.t(ypixel)


def project_point_radial_batch(x, R, T, f, c, k, p, trans):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    x = [_x - _T.reshape(1, 1, 3) for _x, _T in zip(x, T)]
    ypixel = []
    # iterating over batch size
    for _x, _R, _T, _f, _c, _k, _p, _tr in zip(x, R, T, f, c, k, p, trans):
        num_poses = _x.shape[0]
        _x = _x.permute((0, 2, 1))
        _R_repeat = _R[None].repeat(num_poses, 1, 1)
        _x = torch.bmm(_R_repeat, _x)
        y = _x[:, :2] / (_x[:, 2][:, None] + 1e-5)
        r2 = torch.sum(y ** 2, 1, keepdim=True)
        r2exp = torch.cat([r2, r2 ** 2, r2 ** 3], 1)
        radial = 1 + torch.einsum(
            "pij,pij->pj", _k.repeat((r2exp.shape[0], 1, r2exp.shape[-1])), r2exp
        )
        tan = _p[0] * y[:, 1] + _p[1] * y[:, 0]
        corr = (radial[:, None] + 2 * tan[:, None]).repeat((1, 2, 1))
        y = y * corr + torch.bmm(
            torch.cat([_p[1], _p[0]])[None, ..., None].repeat(_x.shape[0], 1, 1), r2
        )
        ypix = (_f.repeat(num_poses, 1, 1) * y) + _c.repeat(num_poses, 1, 1)

        print("ypix shape:", ypix.shape)
        # converting the coordinates into homogeneous form
        ypix = torch.cat(
            (
                ypix,
                torch.ones(
                    ypix.shape[0],
                    1,
                    ypix.shape[-1],
                    device=ypix.device,
                    dtype=ypix.dtype,
                ),
            ),
            1,
        )
        print("ypix shape:", ypix.shape)
        print("transformation shape:", _tr[None].shape)
        print("transformation shape v2:", _tr[None].repeat(ypix.shape[0], 1, 1).shape)
        print("type of _tr", type(_tr))
        # Apply the transformation
        ypix = torch.bmm(_tr[None].repeat(ypix.shape[0], 1, 1), ypix)
        ypixel.append(ypix.permute(0, 2, 1)[..., :2])
    return ypixel


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera, device=x.device)
    return project_point_radial(x, R, T, f, c, k, p)


def project_pose_batch(x, cam, trans):
    R, T, f, c, k, p = cam["R"], cam["T"], cam["f"], cam["c"], cam["k"], cam["p"]
    return project_point_radial_batch(x, R, T, f, c, k, p, trans)

def project_points_radial_OR_4D(input, R, T, f, c, k, p):
    """
    Args
        input: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    normalized_scale = 500
    n = input.shape[0]
    xcam = torch.mm(R.inverse(), torch.t(input / normalized_scale) - T)

    xcam[1, :] *= -1
    xcam[2, :] *= -1

    y = xcam[:2, :] / (xcam[2, :] + 1e-5)

    #v1
    # Calculate radial distortion
    #r2 = y[0, :]**2 + y[1, :]**2
    #radial_distortion = 1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3
    # Apply radial and tangential distortion
    #y_distorted = torch.zeros_like(y)
    #y_distorted[0, :] = y[0, :] * radial_distortion + 2 * p[0] * y[0, :] * y[1, :] + p[1] * (r2 + 2 * y[0, :]**2)
    #y_distorted[1, :] = y[1, :] * radial_distortion + p[0] * (r2 + 2 * y[1, :]**2) + 2 * p[1] * y[0, :] * y[1, :]
    # Convert both distorted and non-distorted points to pixel space
    #ypixel_distorted = (f * y_distorted) + c

    #v2
    kexp = k.repeat((1, n))
    r2 = torch.sum(y ** 2, 0, keepdim=True) / (normalized_scale ** 2)
    r2 = torch.clamp(r2, max=1e10)
    r2exp = torch.cat([r2, r2 ** 2, r2 ** 3], 0)
    radial = 1 + torch.einsum("ij,ij->j", kexp, r2exp)

    tan = p[0] * y[1] + p[1] * y[0] / normalized_scale
    corr = (radial + 2 * tan).repeat((2, 1))

    y_corr = y * corr + torch.ger(torch.cat([p[1], p[0]]).view(-1), r2.view(-1))
    ypixel_distorted = (f * y_corr) + c

    # Calculate the difference between distorted and non-distorted points
    ypixel_no_distortion = (f * y) + c
    difference = ypixel_distorted - ypixel_no_distortion

    # Print the difference
    print("Difference between distorted and non-distorted points:")
    print(k, p)
    print(difference.T)

    return torch.t(ypixel_distorted)

def project_pose_OR_4D(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera, device=x.device)
    return project_points_radial_OR_4D(x, R, T, f, c, k, p)


def project_points_radial_OR_4D_batch(x_list, R, T, f, c, k, p, trans):
    output = []
    for x_tensor in x_list:
        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0)

        for x_batch, _R, _T, _f, _c, _k, _p, _tr in zip(x_tensor, R, T, f, c, k, p, trans):
            ypixel = project_points_radial_OR_4D(x_batch, _R, _T, _f, _c, _k , _p)

            ypixel_homogeneous = torch.cat(
                (ypixel, torch.ones(ypixel.shape[0], 1, device=ypixel.device, dtype=ypixel.dtype)), dim=1
            )
            ypixel_homogeneous = ypixel_homogeneous.transpose(0, 1)
            ypixel_homogeneous = ypixel_homogeneous.reshape(1, ypixel_homogeneous.shape[0], ypixel_homogeneous.shape[1])

            ypixel_transformed = torch.bmm(_tr[None].repeat(ypixel_homogeneous.shape[0], 1, 1), ypixel_homogeneous)
            output.append(ypixel_transformed.permute(0, 2, 1)[..., :2])
    return output


def project_pose_OR_4D_batch(x, camera, trans):
    R, T, f, c, k, p = unfold_camera_param(camera, device=x[0].device)
    return project_points_radial_OR_4D_batch(x, R, T, f, c, k, p, trans)

def world_to_camera_frame(x, R, T):
    """
    Args
        x: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 3d points in camera coordinates
    """

    R = torch.as_tensor(R, device=x.device)
    T = torch.as_tensor(T, device=x.device)
    xcam = torch.mm(R, torch.t(x) - T)
    return torch.t(xcam)


def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """

    R = torch.as_tensor(R, device=x.device)
    T = torch.as_tensor(T, device=x.device)
    xcam = torch.mm(torch.t(R), torch.t(x))
    xcam = xcam + T  # rotate and translate
    return torch.t(xcam)
