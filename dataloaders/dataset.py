import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy import ndimage
from typing import Tuple, Sequence, Dict, Any, Optional
import math
import random
from scipy.stats import poisson

# =========================
# 数据集
# =========================
class BTCV(Dataset):
    """
    Synapse/BTCV：基于 .h5 文件，键为:
        'image' : (W,H,D) 预处理后 CT
        'label' : (W,H,D) 语义标签
        'edge'  : (W,H,D) 预先计算好的 Sobel 语义边界 (0~1, float32)
    """
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
            # 兼容一下没有 edge 的老文件
            edge  = h5f["edge"][:] if "edge" in h5f.keys() else None

        sample = {"image": image, "label": label}
        if edge is not None:
            sample["edge"] = edge

        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop3D:
    """
    随机裁剪 3D patch；不足则 zero-padding。
    输入/输出:
        sample['image']: (W,H,D)
        sample['label']: (W,H,D)
        sample['edge'] : (W,H,D)  —— 若存在则同步裁剪
    """
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image, label = sample["image"], sample["label"]
        edge = sample.get("edge", None)

        # zero-padding（对称 padding 并加 3 个体素余量，避免边界随机采样溢出）
        pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
        ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
        pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
        if pw or ph or pd:
            pad_cfg = [(pw, pw), (ph, ph), (pd, pd)]
            image = np.pad(image, pad_cfg, mode="constant", constant_values=0)
            label = np.pad(label, pad_cfg, mode="constant", constant_values=0)
            if edge is not None:
                edge = np.pad(edge, pad_cfg, mode="constant", constant_values=0.0)

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
        if edge is not None:
            edge = edge[w1:w1 + self.output_size[0],
                        h1:h1 + self.output_size[1],
                        d1:d1 + self.output_size[2]]

        out = {"image": image, "label": label}
        if edge is not None:
            out["edge"] = edge
        return out


class PadIfNeeded3D:
    """
    若样本体素尺寸小于 target_size=(W,H,D)，则做对称 pad；否则不处理。
    image/label/edge: numpy array, shape (W,H,D)
    """
    def __init__(self, target_size, pad_value_img=0.0, pad_value_lbl=0, pad_value_edge=0.0):
        self.tw, self.th, self.td = target_size
        self.pad_value_img = pad_value_img
        self.pad_value_lbl = pad_value_lbl
        self.pad_value_edge = pad_value_edge

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        edge = sample.get("edge", None)

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

        image_p = np.pad(
            image, pad_width,
            mode="constant",
            constant_values=self.pad_value_img
        ).astype(image.dtype)
        label_p = np.pad(
            label, pad_width,
            mode="constant",
            constant_values=self.pad_value_lbl
        ).astype(label.dtype)
        if edge is not None:
            edge_p = np.pad(
                edge, pad_width,
                mode="constant",
                constant_values=self.pad_value_edge
            ).astype(edge.dtype)
        else:
            edge_p = None

        out = {"image": image_p, "label": label_p}
        if edge_p is not None:
            out["edge"] = edge_p
        return out


class CenterCrop3D:
    """
    固定中心裁剪到 target_size=(W,H,D)。
    若输入更大，则居中裁剪；若更小，建议先用 PadIfNeeded3D。
    """
    def __init__(self, target_size):
        self.tw, self.th, self.td = target_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        edge = sample.get("edge", None)

        w, h, d = image.shape
        # 起点（确保不会越界；若目标尺寸==当前尺寸，起点为0）
        sw = max(0, (w - self.tw) // 2)
        sh = max(0, (h - self.th) // 2)
        sd = max(0, (d - self.td) // 2)
        ew, eh, ed = sw + self.tw, sh + self.th, sd + self.td

        image_c = image[sw:ew, sh:eh, sd:ed]
        label_c = label[sw:ew, sh:eh, sd:ed]
        if edge is not None:
            edge_c = edge[sw:ew, sh:eh, sd:ed]
        else:
            edge_c = None

        out = {"image": image_c, "label": label_c}
        if edge_c is not None:
            out["edge"] = edge_c
        return out


class RandomTranslate3D:
    """
    仅平移的 3D 数据增强（同步作用于 image/label/edge）
    - 允许亚像素平移（使用 ndimage.shift）
    - image: 线性插值(order=1)
    - edge : 线性插值(order=1, 适合软边界)
    - label: 最近邻插值(order=0)
    - 平移单位：体素 (voxels)
    """
    def __init__(self,
                 max_trans: Tuple[float, float, float] = (4.0, 4.0, 3.0),
                 p: float = 0.5,
                 mode: str = "nearest",
                 cval_img: float = 0.0,
                 cval_lbl: int = 0,
                 cval_edge: float = 0.0):
        self.max_trans = tuple(float(x) for x in max_trans)
        self.p = float(p)
        self.mode = mode
        self.cval_img = float(cval_img)
        self.cval_lbl = int(cval_lbl)
        self.cval_edge = float(cval_edge)

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
        edge: np.ndarray = sample.get("edge", None)

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
        if edge is not None:
            edge_t = ndimage.shift(
                edge,
                shift=shift,
                order=1,              # 保持软边界
                mode=self.mode,
                cval=self.cval_edge,
                prefilter=True
            )
        else:
            edge_t = None

        out = {
            "image": img_t.astype(np.float32, copy=False),
            "label": lbl_t.astype(lbl.dtype, copy=False)
        }
        if edge_t is not None:
            out["edge"] = edge_t.astype(np.float32, copy=False)
        return out
        

class RandomMixedNoise3D:
    """
    只对 image 加噪声，label / edge 不变。
    """
    def __init__(
        self,
        std_range: Tuple[float, float] = (0.01, 0.05),
        p: float = 0.2,
        noise_type: str = "gaussian",  # 支持 gaussian/poisson/salt_pepper/mixed
        salt_pepper_ratio: float = 0.001,
        poisson_scale: float = 10.0
    ):
        self.std_range = std_range
        self.p = p
        self.noise_type = noise_type.lower()
        self.salt_pepper_ratio = np.clip(salt_pepper_ratio, 0.0, 0.01)
        self.poisson_scale = poisson_scale

        assert self.noise_type in ["gaussian", "poisson", "salt_pepper", "mixed"], \
            f"Invalid noise_type: {self.noise_type}. Choose from ['gaussian', 'poisson', 'salt_pepper', 'mixed']"

    def _generate_gaussian_noise(self, img_shape: Tuple[int, int, int], std: float) -> np.ndarray:
        return np.random.normal(0.0, std, size=img_shape).astype(np.float32)

    def _generate_poisson_noise(self, img: np.ndarray, std: float) -> np.ndarray:
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8) * self.poisson_scale
        noise_poisson = poisson.rvs(mu=img_norm + std * self.poisson_scale, size=img.shape) - img_norm
        noise_poisson = noise_poisson.astype(np.float32) * (std / self.poisson_scale)
        return noise_poisson

    def _generate_salt_pepper_noise(self, img_shape: Tuple[int, int, int], img_min: float, img_max: float) -> np.ndarray:
        noise = np.zeros(img_shape, dtype=np.float32)
        salt_mask = np.random.choice(
            [0, 1], size=img_shape,
            p=[1 - self.salt_pepper_ratio/2, self.salt_pepper_ratio/2]
        )
        pepper_mask = np.random.choice(
            [0, 1], size=img_shape,
            p=[1 - self.salt_pepper_ratio/2, self.salt_pepper_ratio/2]
        )
        noise[salt_mask == 1] = img_max * 0.1
        noise[pepper_mask == 1] = img_min * 0.1
        return noise

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            return sample
        
        img, lbl = sample["image"], sample["label"]
        edge = sample.get("edge", None)

        img = img.astype(np.float32)
        std = random.uniform(*self.std_range)
        img_shape = img.shape
        img_min, img_max = img.min(), img.max()
        noise = np.zeros(img_shape, dtype=np.float32)

        if self.noise_type == "gaussian":
            noise = self._generate_gaussian_noise(img_shape, std)
        elif self.noise_type == "poisson":
            noise = self._generate_poisson_noise(img, std)
        elif self.noise_type == "salt_pepper":
            noise = self._generate_salt_pepper_noise(img_shape, img_min, img_max)
        elif self.noise_type == "mixed":
            noise_types = random.sample(
                ["gaussian", "poisson", "salt_pepper"],
                k=random.choice([1, 2])
            )
            for nt in noise_types:
                if nt == "gaussian":
                    noise += self._generate_gaussian_noise(img_shape, std * 0.7)
                elif nt == "poisson":
                    noise += self._generate_poisson_noise(img, std * 0.7)
                elif nt == "salt_pepper":
                    noise += self._generate_salt_pepper_noise(img_shape, img_min, img_max)

        img = (img + noise).astype(np.float32)

        out = {"image": img, "label": lbl}
        if edge is not None:
            out["edge"] = edge
        return out


class ToTensor3D:
    """
    Numpy -> Torch（3D）
    image: (W,H,D) -> (1,W,H,D) float32
    label: (W,H,D) -> long
    edge : (W,H,D) -> (1,W,H,D) float32 (若存在)
    """
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image, label = sample["image"], sample["label"]
        edge = sample.get("edge", None)

        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image_tensor = torch.from_numpy(image).contiguous()
        label_tensor = torch.from_numpy(label).long().contiguous()

        out = {"image": image_tensor, "label": label_tensor}

        if edge is not None:
            edge = edge.reshape(1, edge.shape[0], edge.shape[1], edge.shape[2]).astype(np.float32)
            edge_tensor = torch.from_numpy(edge).contiguous()
            out["edge"] = edge_tensor

        return out
