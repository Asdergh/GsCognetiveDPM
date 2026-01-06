import torch
import cv2
import numpy as np
import math
from PIL import Image
from dataclasses import (dataclass, field, fields)
from typing import (
    Union, 
    Optional, 
    Tuple, 
    NamedTuple
)
from torch.nn.functional import interpolate
from torchvision.transforms import PILToTensor, Resize, Compose
from torchvision.transforms.functional import pil_to_tensor



COLOR_PALETTE_ = [
0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204, 5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 
8, 255, 51, 255, 6, 82, 143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255, 7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 
112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6, 10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255, 20, 255, 8, 41, 255, 5, 153, 
6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15, 20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255, 
31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163, 0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 
0, 255, 173, 10, 0, 255, 173, 255, 0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0, 31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 
0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0, 194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255, 0, 255, 235, 0, 
8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255, 0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 
255, 0, 163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0, 10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 
0, 41, 255, 0, 255, 204, 41, 0, 255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0, 133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 
92, 0, 255
]

@dataclass
class CameraInfo:
    resolution: Tuple[int]=None
    fovx: float=None
    fovy: float=None
    tanfovx: float=None
    tanfovy: float=None
    znear: float=0.01
    zfar: float=100.0
    Translation: Union[np.ndarray, torch.Tensor]=None
    Quat: Optional[Union[np.ndarray, torch.Tensor]]=None
    Rmat: Optional[Union[np.ndarray, torch.Tensor]]=None
    viewmatrix: Optional[Union[np.ndarray, torch.Tensor]]=None
    projmatrix: Optional[Union[np.ndarray, torch.Tensor]]=None
    K: Optional[Union[np.ndarray, torch.Tensor]]=None
    base_img: Optional[Union[np.ndarray, torch.Tensor]]=None
    depth_mask: Optional[Union[np.ndarray, torch.Tensor]]=None
    segmentation_masks: Optional[Union[np.ndarray, torch.Tensor]]=None
    camera_center: Optional[torch.Tensor]=None


    def __post_init__(self) -> None:

        if self.viewmatrix is None:
            assert (self.Quat is not None and self.Translation is not None), ("cannont initialize viewmatrix for camera")
            R = quat2Rmat(self.Quat, "xyzw")
            self.Rmat = torch.Tensor(R)
            # self.viewmatrix = torch.Tensor(getWorld2View2(R, self.Translation))
            self.viewmatrix = torch.eye(4)
            self.viewmatrix[:3, :3] = self.Rmat
            self.viewmatrix[:3, :3] = self.Translation
        
        if self.viewmatrix is not None:
            self.camera_center = self.viewmatrix[3, :3]
        
        if (self.K is not None 
            and self.resolution is not None):
            self.projmatrix = getProjectionMatrix2(
                znear=self.znear,
                zfar=self.zfar,
                K=self.K,
                W=self.resolution[0], H=self.resolution[1]
            ).transpose(0, 1)
        
        if ((self.fovx is None
            or self.fovy is None)
            and self.K is not None):
            self.fovx = 2.0 * math.atan2(self.resolution[0] / 2.0, self.K[0, 0])
            self.fovy = 2.0 * math.atan2(self.resolution[1] / 2.0, self.K[1, 1])
            

        if (self.tanfovx is None 
            or self.tanfovy is None):
            if (self.fovx is not None
                and self.fovy is not None):
                self.tanfovx = math.tan(self.fovx / 2)
                self.tanfovy = math.tan(self.fovy / 2)

    def to(self, arg: str) -> None:
        fields_to_convert_ = ["translation", 
                              "quat", "Rmat", "viewmatrix", "K",
                              "base_img", "depth_mask", "segmentation_masks"]
        for field in fields(self):
            f_name = field.name
            value = getattr(self, f_name)
            if value is not None:
                if f_name in fields_to_convert_:
                    if arg in ["pt", "tensor"]:
                        if isinstance(value, np.ndarray):
                            setattr(self, f_name, torch.Tensor(value))
                    elif arg in ["np", "array"]:
                        if isinstance(value, torch.Tensor):
                            setattr(self, f_name, value.detach().cpu().numpy())
                    elif arg in ["cpu", "cuda"]:
                        print(f_name)
                        assert isinstance(value, torch.Tensor), (f"can only convert pytorch tensors to {arg}!!!")
                        setattr(self, f_name, value.to(arg))
                    else:
                        raise ValueError("unrecognized arg value !!!")
    
    def get_masks(self, cls_idx: list, colorized: bool=True) -> Union[np.ndarray, torch.Tensor]:
        if self.segmentation_masks is not None:
            if len(cls_idx) != 1:
                masks = torch.stack([
                    (
                        (self.segmentation_masks[..., idx] > 0.5) * idx
                        if colorized
                        else self.segmentation_masks[..., idx]
                    )  for idx in cls_idx
                ], dim=-1)
                return masks
            else:
                mask = (
                    (self.segmentation_masks[..., cls_idx[0]] > 0.5) * cls_idx[0]
                    if colorized
                    else self.segmentation_masks[..., cls_idx[0]]
                )
                return mask
        else:
            raise ValueError("specify segmentation masks !!!")
    
    def get_masked_img(self, cls_idx: list) -> Union[np.ndarray, torch.Tensor]:
        masks = self.get_masks(cls_idx)
        masked_img = apply_masks(self.base_img, masks)
        return masked_img

    def set_projection_bounds(self, near: float, far: float) -> None:
        self.znear = near
        self.zfar = far
        self.projmatrix = getProjectionMatrix2(
            znear=self.znear,
            zfar=self.zfar,
            K=self.K,
            W=self.resolution[0], H=self.resolution[1]
        ).transpose(0, 1)
    


def apply_masks(
    gt_img: torch.Tensor,
    masks: torch.Tensor,
    alpha: Optional[float]=0.5,
    format: Optional[str]="WHC"
): 
    if format == "CWH":
        gt_img = gt_img.permute(2, 0, 1)
        

    gt_img = (gt_img.detach().cpu().numpy() * 255.0).astype("uint8")
    gt_img = Image.fromarray(gt_img)

    masks = masks.detach().cpu().numpy()
    masks = (np.argmax(masks, axis=-1) if masks.ndim == 3 else masks).astype("uint8")
    masks = Image.fromarray(masks)
    masks.putpalette(COLOR_PALETTE_)
    
    masked_img = Image.blend(gt_img, masks.convert("RGB"), alpha)
    masked_img = pil_to_tensor(masked_img)
    masked_img = (masked_img / 255.0).permute(1, 2, 0)

    return (masked_img if format == "WHC" else masked_img.permute(1, 2, 0))


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

    P = torch.zeros(4, 4)
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
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


if __name__ == "__main__":

    test = np.random.rand(10000, 3, 3)
    quats = Rmat2quat(test, "xyzw")
    print(quats.shape)


    import matplotlib.pyplot as plt
    from transformers import (DPTImageProcessor, DPTForSemanticSegmentation)
    img = "/media/ram/T71/360_v2/garden/images_8/DSC07965.JPG"
    img = Image.open(img)
    tf = Compose([
        PILToTensor(),
        Resize((224, 224))
    ])
    img = tf(img)
    
    model_v = "Intel/dpt-large-ade"
    preprocessor = DPTImageProcessor.from_pretrained(model_v)
    model = DPTForSemanticSegmentation.from_pretrained(model_v)

    img_prep = preprocessor(img, return_tensors="pt")
    preds = model(**img_prep).logits
    preds = interpolate(preds, size=(224, 224)).squeeze().permute(1, 2, 0)
    
    camera = CameraInfo(
        base_img=img.permute(1, 2, 0),
        segmentation_masks=preds
    )
    camera.to("pt")
    masks = camera.get_masks([8, 19, 3], colorized=False)
    masked_img = camera.get_masked_img([8, 19, 3])
    _, axis = plt.subplots(ncols=4)
    axis[0].imshow(masked_img)
    axis[1].imshow(masks[..., 0].detach())
    axis[2].imshow(masks[..., 1].detach())
    axis[3].imshow(masks[..., 2].detach())
    plt.show()


    
    