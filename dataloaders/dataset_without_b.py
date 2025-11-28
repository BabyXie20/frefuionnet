# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy import ndimage
from typing import Tuple, Sequence, Dict, Any, Optional
import math
import random


# =========================
# 数据集
# =========================
class BTCV(Dataset):
    """Synapse/BTCV：基于 .h5 文件，键为 'image' 与 'label'，形状 (W,H,D)。"""
    def __init__(self, image_list, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = image_list
        print(f"Total {len(self.image_list)} samples line1:train line2:test")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = f"{self._base_dir}/{image_name}.h5"
        with h5py.File(image_path, "r") as h5f:
            image = h5f["image"][:]  # (W,H,D) 预处理后数据
            label = h5f["label"][:]  # (W,H,D)
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample


# =========================
# 基础操作（仅保留裁剪与张量化）
# =========================
class RandomCrop3D:
    """
    随机裁剪 3D patch；不足则 zero-padding。
    输入/输出:
        sample['image']: (W,H,D)
        sample['label']: (W,H,D)
    """
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image, label = sample["image"], sample["label"]

        # zero-padding（对称 padding 并加 3 个体素余量，避免边界随机采样溢出）
        pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
        ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
        pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
        if pw or ph or pd:
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode="constant", constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode="constant", constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        image = image[w1:w1 + self.output_size[0],
                      h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0],
                      h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]
        return {"image": image, "label": label}


class PadIfNeeded3D:
    """
    若样本体素尺寸小于 target_size=(W,H,D)，则做对称 pad；否则不处理。
    image/label: numpy array, shape (W,H,D)
    """
    def __init__(self, target_size, pad_value_img=0.0, pad_value_lbl=0):
        self.tw, self.th, self.td = target_size
        self.pad_value_img = pad_value_img
        self.pad_value_lbl = pad_value_lbl

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        w, h, d = image.shape
        pw = max(0, self.tw - w)
        ph = max(0, self.th - h)
        pd = max(0, self.td - d)
        if pw == 0 and ph == 0 and pd == 0:
            return sample

        # 对称 pad：左⌊/右⌈
        pad_w_left  = pw // 2
        pad_w_right = pw - pad_w_left
        pad_h_top   = ph // 2
        pad_h_bottom= ph - pad_h_top
        pad_d_front = pd // 2
        pad_d_back  = pd - pad_d_front

        pad_width = ((pad_w_left, pad_w_right),
                     (pad_h_top,  pad_h_bottom),
                     (pad_d_front, pad_d_back))

        image_p = np.pad(image, pad_width, mode="constant", constant_values=self.pad_value_img).astype(image.dtype)
        label_p = np.pad(label, pad_width, mode="constant", constant_values=self.pad_value_lbl).astype(label.dtype)

        return {"image": image_p, "label": label_p}


class CenterCrop3D:
    """
    固定中心裁剪到 target_size=(W,H,D)。若输入更大，则居中裁剪；若更小，建议先用 PadIfNeeded3D。
    """
    def __init__(self, target_size):
        self.tw, self.th, self.td = target_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        w, h, d = image.shape
        # 起点（确保不会越界；若目标尺寸==当前尺寸，起点为0）
        sw = max(0, (w - self.tw) // 2)
        sh = max(0, (h - self.th) // 2)
        sd = max(0, (d - self.td) // 2)
        ew, eh, ed = sw + self.tw, sh + self.th, sd + self.td
        image_c = image[sw:ew, sh:eh, sd:ed]
        label_c = label[sw:ew, sh:eh, sd:ed]
        return {"image": image_c, "label": label_c}

class ToTensor3D:
    """
    Numpy -> Torch（3D）
    image: (W,H,D) -> (1,W,H,D) float32
    label: long
    """
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image, label = sample["image"], sample["label"]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image_tensor = torch.from_numpy(image).contiguous()
        label_tensor = torch.from_numpy(label).long().contiguous()
        return {"image": image_tensor, "label": label_tensor}


# =========================
# 几何增强（同步作用于 image/label）
# =========================

class RandomTranslate3D:
    """
    仅平移的 3D 数据增强（同步作用于 image/label）
    - 允许亚像素平移（使用 ndimage.shift）
    - image: 线性插值(order=1)
    - label: 最近邻插值(order=0)
    - 平移单位：体素 (voxels)

    Args:
        max_trans: (tx, ty, tz) 三轴的最大绝对平移幅度（体素）。实际平移会在 [-tx, tx] 等范围内均匀采样。
        p: 触发概率
        mode: 边界模式（'nearest'/'reflect'/'constant' 等，参考 scipy.ndimage）
        cval_img: image 常量填充值（当 mode='constant' 时有效）
        cval_lbl: label 常量填充值（当 mode='constant' 时有效，一般为 0）
    """
    def __init__(self,
                 max_trans: Tuple[float, float, float] = (4.0, 4.0, 3.0),
                 p: float = 0.5,
                 mode: str = "nearest",
                 cval_img: float = 0.0,
                 cval_lbl: int = 0):
        self.max_trans = tuple(float(x) for x in max_trans)
        self.p = float(p)
        self.mode = mode
        self.cval_img = float(cval_img)
        self.cval_lbl = int(cval_lbl)

    def _rand_shift(self) -> Tuple[float, float, float]:
        tx = random.uniform(-self.max_trans[0], self.max_trans[0])
        ty = random.uniform(-self.max_trans[1], self.max_trans[1])
        tz = random.uniform(-self.max_trans[2], self.max_trans[2])
        return (tx, ty, tz)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample

        img: np.ndarray = sample["image"]
        lbl: np.ndarray = sample["label"]

        # 注意：ndimage.shift 的 shift 顺序与数组轴一致，这里假设 img.shape = (W, H, D)
        shift = self._rand_shift()

        img_t = ndimage.shift(
            img,
            shift=shift,
            order=1,                  # 线性插值
            mode=self.mode,
            cval=self.cval_img,
            prefilter=True
        )
        lbl_t = ndimage.shift(
            lbl,
            shift=shift,
            order=0,                  # 最近邻，不引入新标签
            mode=self.mode,
            cval=self.cval_lbl,
            prefilter=False
        )
        return {"image": img_t.astype(np.float32, copy=False),
                "label": lbl_t.astype(lbl.dtype, copy=False)}
        
class RandomFlip3D:
    """随机沿给定轴翻转；对每个轴独立以概率 p 翻转。"""
    def __init__(self, axes: Sequence[int] = (0, 1, 2), p: float = 0.5):
        self.axes = tuple(axes)
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img, lbl = sample["image"], sample["label"]
        for ax in self.axes:
            if random.random() < self.p:
                img = np.flip(img, axis=ax).copy()
                lbl = np.flip(lbl, axis=ax).copy()
        return {"image": img, "label": lbl}


class RandomRotate90_3D:
    """随机 90° 离散旋转（在 (0,1)、(0,2)、(1,2) 中随机选一对轴，旋转 90/180/270 度）。"""
    def __init__(self, p: float = 0.5):
        self.p = p
        self.axis_pairs = [(0, 1), (0, 2), (1, 2)]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        axes = random.choice(self.axis_pairs)
        k = random.randint(1, 3)
        img = np.rot90(img, k=k, axes=axes).copy()
        lbl = np.rot90(lbl, k=k, axes=axes).copy()
        return {"image": img, "label": lbl}


class RandomAffine3D:
    """
    小角度旋转 + 等比缩放 + 平移（体素单位）
    - image: 线性插值
    - label: 最近邻
    """
    def __init__(self,
                 rot_deg: Tuple[float, float, float] = (10.0, 10.0, 10.0),
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 translate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 p: float = 0.2,
                 mode: str = "nearest"):
        self.rot_deg = rot_deg
        self.scale_range = scale_range
        self.translate = translate
        self.p = p
        self.mode = mode

    @staticmethod
    def _rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]], dtype=np.float32)
        Ry = np.array([[ cy, 0, sy],
                       [ 0, 1,  0],
                       [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]], dtype=np.float32)
        return Rx @ Ry @ Rz

    @staticmethod
    def _invert_affine(M: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M_inv = np.linalg.inv(M)
        t_inv = - M_inv @ t
        return M_inv.astype(np.float32), t_inv.astype(np.float32)

    @staticmethod
    def _apply(img: np.ndarray, lbl: np.ndarray,
               matrix: np.ndarray, offset: Sequence[float],
               order_img: int, order_lbl: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        img_warp = ndimage.affine_transform(
            img, matrix=matrix, offset=offset, order=order_img, mode=mode, cval=0.0, prefilter=(order_img > 1)
        )
        lbl_warp = ndimage.affine_transform(
            lbl, matrix=matrix, offset=offset, order=order_lbl, mode=mode, cval=0, prefilter=False
        )
        return img_warp, lbl_warp

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample

        img, lbl = sample["image"], sample["label"]

        # 随机角度（度 -> 弧度）
        rx = np.deg2rad(random.uniform(-self.rot_deg[0], self.rot_deg[0]))
        ry = np.deg2rad(random.uniform(-self.rot_deg[1], self.rot_deg[1]))
        rz = np.deg2rad(random.uniform(-self.rot_deg[2], self.rot_deg[2]))
        R = self._rotation_matrix_xyz(rx, ry, rz)

        # 等比缩放
        s = random.uniform(*self.scale_range)
        S = np.eye(3, dtype=np.float32) * s

        # 线性部分
        M = R @ S  # x' = M x + t

        # 平移
        t = np.array([random.uniform(-self.translate[0], self.translate[0]),
                      random.uniform(-self.translate[1], self.translate[1]),
                      random.uniform(-self.translate[2], self.translate[2])], dtype=np.float32)

        # 以体素中心为枢轴
        c = (np.array(img.shape, dtype=np.float32) - 1.0) / 2.0
        t_total = c - M @ c + t  # x' = Mx + (c - M c + t)

        # 传给 scipy 需要逆映射
        M_inv, t_inv = self._invert_affine(M, t_total)
        img_w, lbl_w = self._apply(img, lbl, M_inv, t_inv, order_img=1, order_lbl=0, mode=self.mode)
        return {"image": img_w, "label": lbl_w}


class RandomElasticDeform3D:
    """
    3D 弹性形变（位移场 + 高斯平滑）
    - image: 线性插值
    - label: 最近邻
    """
    def __init__(self,
                 alpha: float = 8.0,
                 sigma: float = 10.0,
                 p: float = 0.15,
                 mode: str = "nearest",
                 rng: Optional[np.random.RandomState] = None):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.mode = mode
        self.rng = rng

    @staticmethod
    def _gen_displacement(shape, sigma, alpha, rng: Optional[np.random.RandomState]):
        if rng is None:
            rng = np.random.RandomState(None)
        disp = rng.rand(*shape).astype(np.float32) * 2 - 1
        disp = ndimage.gaussian_filter(disp, sigma=sigma, mode="reflect", cval=0) * alpha
        return disp

    @staticmethod
    def _warp(img: np.ndarray, lbl: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              mode: str) -> Tuple[np.ndarray, np.ndarray]:
        w, h, d = img.shape
        grid_w, grid_h, grid_d = np.meshgrid(np.arange(w), np.arange(h), np.arange(d), indexing="ij")
        map_w = grid_w + dx
        map_h = grid_h + dy
        map_d = grid_d + dz
        coords = np.vstack([map_w.reshape(-1), map_h.reshape(-1), map_d.reshape(-1)])
        img_def = ndimage.map_coordinates(img, coords, order=1, mode=mode).reshape(img.shape)
        lbl_def = ndimage.map_coordinates(lbl, coords, order=0, mode=mode).reshape(lbl.shape)
        return img_def, lbl_def

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        shape = img.shape
        dx = self._gen_displacement(shape, self.sigma, self.alpha, self.rng)
        dy = self._gen_displacement(shape, self.sigma, self.alpha, self.rng)
        dz = self._gen_displacement(shape, self.sigma, self.alpha, self.rng)
        img_def, lbl_def = self._warp(img, lbl, dx, dy, dz, self.mode)
        return {"image": img_def, "label": lbl_def}


# =========================
# 强度增强（仅作用 image）
# =========================
class RandomGaussianBlur3D:
    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.2), p: float = 0.2):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        sigma = random.uniform(*self.sigma_range)
        img = ndimage.gaussian_filter(img, sigma=sigma)
        return {"image": img, "label": lbl}


class RandomGaussianNoise3D:
    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.05), p: float = 0.2):
        self.std_range = std_range
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0.0, std, size=img.shape).astype(np.float32)
        img = (img + noise).astype(np.float32)
        return {"image": img, "label": lbl}


class RandomBrightnessContrast3D:
    """I' = a * I + b"""
    def __init__(self,
                 brightness_range: Tuple[float, float] = (-0.1, 0.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 p: float = 0.3):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        a = random.uniform(*self.contrast_range)
        b = random.uniform(*self.brightness_range)
        img = (a * img + b).astype(np.float32)
        return {"image": img, "label": lbl}


class RandomGamma3D:
    """Gamma: 先将强度拉到 [0,1]（按当前 patch 的 min/max），做 I^gamma 再映回原范围。"""
    def __init__(self,
                 gamma_range: Tuple[float, float] = (0.7, 1.5),
                 clip_min: Optional[float] = None,
                 clip_max: Optional[float] = None,
                 p: float = 0.3):
        self.gamma_range = gamma_range
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        gamma = random.uniform(*self.gamma_range)
        x = img.copy()
        if self.clip_min is not None and self.clip_max is not None:
            x = np.clip(x, self.clip_min, self.clip_max)
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            x_norm = (x - xmin) / (xmax - xmin)
            x_aug = np.power(x_norm, gamma)
            x = x_aug * (xmax - xmin) + xmin
        return {"image": x.astype(np.float32), "label": lbl}


class RandomIntensityScaleShift3D:
    """I' = I * s + t"""
    def __init__(self,
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 shift_range: Tuple[float, float] = (-0.05, 0.05),
                 p: float = 0.25):
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        s = random.uniform(*self.scale_range)
        t = random.uniform(*self.shift_range)
        img = (img * s + t).astype(np.float32)
        return {"image": img, "label": lbl}
