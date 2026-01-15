import torch
import cv2
import numpy as np
import math
from PIL import Image
from dataclasses import (dataclass, field, fields)
from torch.nn.functional import interpolate
from torchvision.transforms import PILToTensor, Resize, Compose
from torchvision.transforms.functional import pil_to_tensor
from sklearn.metrics import pairwise_distances
from ..types import *


def quat2Rmat(q: Union[torch.Tensor, np.ndarray], in_format="xyzw") -> np.ndarray:
    
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()    

    if (q.ndim == 1):
        batched = False
        N = 1
        q = q.reshape(1, 4)
    elif (q.ndim == 2 and q.shape[-1] == 4):
        N = q.shape[0]
        batched = True
    else:
        raise ValueError(f"q is not quaternion vector or batch of vectors. Size: {q.shape}")
        
    if in_format == "xyzw":
        x = q[:, 0]
        y = q[:, 1]
        z = q[:, 2]
        w = q[:, 3]
    else:
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        w = q[:, 0]
    
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    Rmat_batched = np.zeros((N, 3, 3))
    Rmat_batched[:, 0, 0] = 1 - 2*(y2 + z2)
    Rmat_batched[:, 0, 1] = 2*(xy - wz)
    Rmat_batched[:, 0, 2] = 2*(xz + wy)

    Rmat_batched[:, 1, 0] = 2*(xy + wz)
    Rmat_batched[:, 1, 1] = 1 - 2*(x2 + z2)
    Rmat_batched[:, 1, 2] = 2*(yz - wx)

    Rmat_batched[:, 2, 0] = 2*(xz - wy)
    Rmat_batched[:, 2, 1] = 2*(yz + wx)
    Rmat_batched[:, 2, 2] = 1 - 2*(x2 + y2)

    return (Rmat_batched[0] if not batched else Rmat_batched)

def Rmat2quat(Rmat: Union[torch.Tensor, np.ndarray], out_format="xyzw") -> np.ndarray:
    
    if isinstance(Rmat, torch.Tensor):
        Rmat = Rmat.detach().cpu().numpy() 

    if Rmat.ndim == 3:
        batched = True
        N = Rmat.shape[0]
    else:
        N = 1
        batched = False
        Rmat = Rmat.reshape(1, 3, 3)

    Tr = np.diagonal(Rmat, axis1=1, axis2=2).sum(axis=1)
    q_result = np.zeros((N, 4))
    max_diag = np.stack([Rmat[:, 0, 0], Rmat[:, 1, 1], Rmat[:, 2, 2]], axis=-1).max(axis=1)
    
    where_tr_pos = (Tr > 0)
    if where_tr_pos.sum() != 0.0:
        w = np.sqrt(1 + Tr) / 2
        x = (Rmat[where_tr_pos, 2, 1] - Rmat[where_tr_pos, 1, 2]) / (4 * w + 1e-6)
        y = (Rmat[where_tr_pos, 0, 2] - Rmat[where_tr_pos, 2, 0]) / (4 * w + 1e-6)
        z = (Rmat[where_tr_pos, 1, 0] - Rmat[where_tr_pos, 0, 1]) / (4 * w + 1e-6)
        q_result[where_tr_pos] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)
    
    where_r11_max = (Rmat[:, 0, 0] == max_diag)
    if where_r11_max.sum() != 0.0:
        x = (np.sqrt(1 + 2 * Rmat[where_r11_max, 0, 0] - Tr[where_r11_max])) / 2
        w = (Rmat[where_r11_max, 2, 1] - Rmat[where_r11_max, 1, 2]) / (4 * x + 1e-6)
        y = (Rmat[where_r11_max, 0, 1] + Rmat[where_r11_max, 1, 0]) / (4 * x + 1e-6)
        z = (Rmat[where_r11_max, 0, 2] + Rmat[where_r11_max, 2, 0]) / (4 * x + 1e-6)
        q_result[where_r11_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)

    where_r22_max = (Rmat[:, 1, 1] == max_diag)
    if where_r22_max.sum() != 0.0:
        y = (np.sqrt(1 + 2 * Rmat[where_r22_max, 1, 1] - Tr[where_r22_max])) / 2
        w = (Rmat[where_r22_max, 0, 2] - Rmat[where_r22_max, 2, 0]) / (4 * y + 1e-6)
        x = (Rmat[where_r22_max, 0, 1] + Rmat[where_r22_max, 1, 0]) / (4 * y + 1e-6)
        z = (Rmat[where_r22_max, 1, 2] + Rmat[where_r22_max, 2, 1]) / (4 * y + 1e-6)
        q_result[where_r22_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)

    where_r33_max = (Rmat[:, 2, 2] == max_diag)
    if where_r33_max.sum() != 0.0:
        z = (np.sqrt(1 + 2 * Rmat[where_r33_max, 2, 2] - Tr[where_r33_max])) / 2
        w = (Rmat[where_r33_max, 1, 0] - Rmat[where_r33_max, 0, 1]) / (4 * z + 1e-6)
        x = (Rmat[where_r33_max, 0, 2] - Rmat[where_r33_max, 2, 0]) / (4 * z + 1e-6)
        y = (Rmat[where_r33_max, 1, 2] - Rmat[where_r33_max, 2, 1]) / (4 * z + 1e-6)
        q_result[where_r33_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)
    
    q_norms = np.linalg.norm(q_result, axis=-1, keepdims=True)
    q_result = (q_result / q_norms)
    return (q_result[0] if not batched else q_result)



def getProjectionMatrix2(znear, zfar, K, W, H):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    top = znear * cy / fy
    bottom = -znear * (H - cy) / fy
    right = znear * (W - cx) / fx
    left = -znear * cx / fx

    P = np.eye(4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = (t + translate) * scale
    w2c = np.linalg.inv(Rt)
    w2c = w2c @ np.diag([1, -1, -1, 1])
    return w2c
    

def ColmapNnerf_convertion(
    input: Union[torch.Tensor, np.ndarray], 
    in_type: str, 
    in_mode: Optional[str]=None,
    permute_order: Tuple[int, int]=None
): 
      
    P = np.eye(3)
    (i, j) = permute_order
    if permute_order is not None:
        P[[i, j]] = P[[j, i]]

    reorder_Mat = np.diag([1, -1, -1]) 
    convertion_Mat = reorder_Mat @ P
    if in_type in ["points", "points3D", "pts"]:
        result = (input @ convertion_Mat[:3, :3])
    elif in_type in ["view", "viewmat", "viewmatrix"]:
        if in_mode is not None:
            if in_mode == "c2w":

                R = input[:3, :3] @ reorder_Mat
                R_inv = R.T
                t_inv = -(R_inv @ input[:3, 3])

                result = np.eye(4)
                result[:3, :3] = R_inv
                result[:3, 3] = t_inv

            elif in_mode == "w2c":
                input = (convertion_Mat @ input)
                result = (np.linalg.inv(input) @ convertion_Mat)
        else:
            result = convertion_Mat @ input
    
    return result

def get_camera_extent(
    camcents: Union[np.ndarray, torch.Tensor], 
    method: str="bounding-sphere",
    percentile: Optional[float]=90
):
    if isinstance(camcents, torch.Tensor):
        camcents = camcents.detach().cpu().numpy()
    
    N = camcents.shape[0]
    if method == "bounding-sphere":
        centroid = np.median(camcents, axis=0)
        dists = np.linalg.norm(centroid - camcents)
        radius = np.percentile(dists, percentile)
        cam_extent = 2.0 * radius
    
    if method == "bbox-diag":
        max_camcent = np.percentile(camcents, 100)
        min_camcent = np.percentile(camcents, 0)
        cam_extent = np.linalg.norm(max_camcent - min_camcent)
    
    if method == "max-distance":
        dists = pairwise_distances(camcents, metric="euclidian")
        cam_extent = np.max(dists)
    
    return cam_extent



def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene.
    if center_method == "focus":
        # find the closest point to the origin for each camera's center ray
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # use center of the camera positions
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def align_principal_axes(point_cloud):
    # Compute centroid
    centroid = np.median(point_cloud, axis=0)

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
    # is the principal axis with the smallest eigenvalue.
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant of the eigenvectors is
    # negative, then we need to flip the sign of one of the eigenvectors.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.T

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform

def transform_points(matrix, points):
    """Transform points using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        points: Nx3 array of points

    Returns:
        Nx3 array of transformed points
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds