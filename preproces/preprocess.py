import os
import nibabel as nib
import SimpleITK as sitk
import h5py
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 3D 语义边界提取函数
# =========================
def semantic_edge_3d_union(label_np, ignore_bg=True):
    """
    根据 3D 语义标签生成“语义边界”:
    只要一个体素与任意一个 6 邻域体素的类别不同，就认为是边界。

    label_np: (X, Y, Z) int 标签 (0 ~ num_classes-1)
    返回:
        edge_np: (X, Y, Z) uint8, 0/1 边界图
    """
    edge = np.zeros_like(label_np, dtype=np.uint8)

    # 分别在 X/Y/Z 三个轴上与前后体素比较
    for axis in range(3):
        diff_forward = label_np != np.roll(label_np, -1, axis=axis)
        diff_backward = label_np != np.roll(label_np, 1, axis=axis)
        edge |= (diff_forward | diff_backward).astype(np.uint8)

    if ignore_bg:
        # 可选：将背景内部的“边界”置零，只保留器官一侧
        edge[label_np == 0] = 0

    return edge


# =========================
# 可视化：Image / Label / Edge
# =========================
def visualize_slices(img_np, label_np, edge_np, case_idx, output_folder):
    """
    可视化三个正交切片：
        每一行：Sagittal / Coronal / Axial
        每一列：Image / Label / Edge
    """
    W, H, D = img_np.shape
    w_mid = W // 2
    h_mid = H // 2
    d_mid = D // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'Case{case_idx:04d} - Image / Label / Edge', fontsize=16)

    slice_indices = [
        (w_mid, ':', ':'),  # Sagittal: X 固定
        (':', h_mid, ':'),  # Coronal : Y 固定
        (':', ':', d_mid),  # Axial   : Z 固定
    ]
    titles = ['Sagittal', 'Coronal', 'Axial']

    for i, (sx, sy, sz) in enumerate(slice_indices):
        # Python 的 slice 不能直接用 ':' 字符，这里用 eval 处理
        x_idx = eval(str(sx))
        y_idx = eval(str(sy))
        z_idx = eval(str(sz))

        img_slice   = img_np[x_idx, y_idx, z_idx]
        label_slice = label_np[x_idx, y_idx, z_idx]
        edge_slice  = edge_np[x_idx, y_idx, z_idx]

        # Image
        ax_img = axes[i, 0]
        im0 = ax_img.imshow(img_slice, cmap='gray', aspect='auto')
        ax_img.set_title(f'{titles[i]} - Image')
        ax_img.axis('off')

        # Label
        ax_lbl = axes[i, 1]
        im1 = ax_lbl.imshow(label_slice, cmap=plt.cm.tab20, aspect='auto')
        ax_lbl.set_title(f'{titles[i]} - Label')
        ax_lbl.axis('off')
        plt.colorbar(im1, ax=ax_lbl, shrink=0.6)

        # Edge
        ax_edge = axes[i, 2]
        im2 = ax_edge.imshow(edge_slice, cmap='hot', vmin=0, vmax=1, aspect='auto')
        ax_edge.set_title(f'{titles[i]} - Edge (semantic)')
        ax_edge.axis('off')
        plt.colorbar(im2, ax=ax_edge, shrink=0.6)

    vis_save_path = os.path.join(output_folder, f'{case_idx:04d}_img_label_edge.png')
    plt.tight_layout()
    plt.savefig(vis_save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =========================
# 单个 Case 预处理
# =========================
def preprocess_single_case(image_path, label_path, output_h5_path, case_idx, output_folder):
    # 1. 读取原始 NIfTI
    img_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    img_data = img_nii.get_fdata()
    label_data = img_nii.__class__.from_filename(label_path).get_fdata()  # 或直接 label_nii.get_fdata()
    print(f"Case{case_idx:04d} - Original shape: image{img_data.shape}, label{label_data.shape}")

    # 2. HU 截断
    img_clipped = np.clip(img_data, -125, 275)

    # 3. 转 SimpleITK 以重采样
    img_spacing = tuple(map(float, img_nii.header.get_zooms()))
    label_spacing = tuple(map(float, label_nii.header.get_zooms()))
    assert img_spacing == label_spacing, f"Spacing mismatch: {img_spacing} vs {label_spacing}"
    direction_matrix = np.eye(3).flatten()

    # nibabel: (X,Y,Z)；SimpleITK: (Z,Y,X)，所以转置
    img_sitk = sitk.GetImageFromArray(img_clipped.transpose(2, 1, 0))
    img_sitk.SetSpacing(img_spacing)
    img_sitk.SetOrigin(tuple(img_nii.affine[:3, 3]))
    img_sitk.SetDirection(direction_matrix)

    label_array = label_data.astype(np.uint8)
    label_sitk = sitk.GetImageFromArray(label_array.transpose(2, 1, 0))
    label_sitk.SetSpacing(label_spacing)
    label_sitk.SetOrigin(tuple(label_nii.affine[:3, 3]))
    label_sitk.SetDirection(direction_matrix)

    # 4. 计算重采样目标 spacing & size
    target_spacing = (1.5, 1.5, 2.0)  # (sx, sy, sz)
    original_physical_size = [img_data.shape[i] * img_spacing[i] for i in range(3)]
    new_size = [int(round(original_physical_size[i] / target_spacing[i])) for i in range(3)]

    # 5. 重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(direction_matrix)
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    # 图像用线性插值
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_img = resampler.Execute(img_sitk)

    # 标签用最近邻插值
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_label = resampler.Execute(label_sitk)

    # 6. 转回 numpy，恢复为 (X,Y,Z)，并归一化
    img_np = sitk.GetArrayFromImage(resampled_img).transpose(2, 1, 0)   # (X,Y,Z)
    label_np = sitk.GetArrayFromImage(resampled_label).transpose(2, 1, 0)

    img_min, img_max = img_np.min(), img_np.max()
    img_np = (img_np - img_min) / (img_max - img_min + 1e-8)

    # 7. 计算 3D 语义边界
    edge_np = semantic_edge_3d_union(label_np, ignore_bg=True)  # (X,Y,Z) 0/1

    # 8. 保存 H5：image / label / edge
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('image', data=img_np.astype(np.float32))
        f.create_dataset('label', data=label_np.astype(np.uint8))
        f.create_dataset('edge',  data=edge_np.astype(np.uint8))

    # 9. 可视化 Image / Label / Edge
    visualize_slices(img_np, label_np, edge_np, case_idx, output_folder)
    print(f"Case{case_idx:04d} - Processed and saved to: {output_h5_path}")


# =========================
# 批量预处理
# =========================
def batch_preprocess(input_images_folder, input_labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # BTCV training cases: 1–10 + 21–40
    case_indices = list(range(1, 11)) + list(range(21, 41))

    for case_idx in case_indices:
        img_name = f'img{case_idx:04d}.nii.gz'
        label_name = f'label{case_idx:04d}.nii.gz'
        img_path = os.path.join(input_images_folder, img_name)
        label_path = os.path.join(input_labels_folder, label_name)

        if os.path.exists(img_path) and os.path.exists(label_path):
            output_h5 = os.path.join(output_folder, f'{case_idx:04d}.h5')
            preprocess_single_case(img_path, label_path, output_h5, case_idx, output_folder)
        else:
            print(f"Case{case_idx:04d} - Warning: file not found ({img_name} or {label_name})")


# =========================
# main
# =========================
if __name__ == "__main__":
    input_images_folder = "../data/btcv/imagesTr"
    input_labels_folder = "../data/btcv/labelsTr"
    output_folder = "../data1/btcv_h5"

    batch_preprocess(input_images_folder, input_labels_folder, output_folder)
