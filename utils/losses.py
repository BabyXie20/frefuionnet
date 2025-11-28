import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10
        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=True, weighted_pixel_map=None):
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i], weighted_pixel_map)
            loss += dice_loss * weight[i]

        return loss / self.n_classes

class CrossEntropy3D(nn.Module):
    def __init__(self, class_weight=None, ignore_index=-1):
        """
        class_weight: list 或 1D tensor, 长度 = n_classes，用于类别不平衡
        ignore_index: 标签中要忽略的类别ID（例如背景不用就可以设为 -1）
        """
        super().__init__()
        if class_weight is not None:
            self.register_buffer('class_weight',
                                 torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None
        self.ignore_index = ignore_index

    def forward(self, inputs, target, weighted_pixel_map=None):
        """
        inputs: [B, C, D, H, W] —— 网络的logits（不要先softmax）
        target: [B, D, H, W] 或 [B, 1, D, H, W] —— 整型类别id
        weighted_pixel_map: [B, D, H, W] 或 [B, 1, D, H, W] —— 像素权重图(可选)
        """
        if target.dim() == inputs.dim():      # [B,1,D,H,W]
            target_ce = target.squeeze(1).long()
        else:
            target_ce = target.long()

        ce = F.cross_entropy(
            inputs,
            target_ce,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction='none'         # 先不做平均，后面自己权重
        )                            # ce: [B, D, H, W]

        # ignore_index 对应的位置权重设为0
        if self.ignore_index >= 0:
            valid_mask = (target_ce != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(ce)

        if weighted_pixel_map is not None:
            if weighted_pixel_map.dim() == ce.dim() + 1:
                # [B,1,D,H,W] -> [B,D,H,W]
                weighted_pixel_map = weighted_pixel_map.squeeze(1)
            pixel_weight = weighted_pixel_map * valid_mask
        else:
            pixel_weight = valid_mask

        ce = (ce * pixel_weight).sum() / (pixel_weight.sum() + 1e-10)
        return ce
    

def compute_soft_edge_loss(edge_logits: torch.Tensor,
                           edge_gt: torch.Tensor,
                           alpha: float = 0.5,
                           smooth: float = 1e-6) -> torch.Tensor:
    """
    计算 Sobel 软边界的监督损失。
    
    Args:
        edge_logits: [B, 1, D, H, W] 或 [B, 1, W, H, D]，网络预测的边界 logits（未过 sigmoid）
        edge_gt    : [B, 1, D, H, W] / [B, D, H, W]，Sobel 生成的软边界 (0~1, float)
        alpha      : L1 与 Dice 的权重，loss = alpha * L1 + (1 - alpha) * Dice
        smooth     : Dice 平滑项，防止除零f
        
    Returns:
        标量 loss（torch.Tensor）
    """
    # 保证 edge_gt 形状与 edge_logits 对齐
    if edge_gt.dim() == 4:
        # [B,D,H,W] -> [B,1,D,H,W]
        edge_gt = edge_gt.unsqueeze(1)
    edge_gt = edge_gt.float()

    if edge_logits.shape != edge_gt.shape:
        raise ValueError(f"shape mismatch: edge_logits {edge_logits.shape} vs edge_gt {edge_gt.shape}")

    # 1) sigmoid 概率
    pred = torch.sigmoid(edge_logits)  # [B,1,...]

    # 2) L1 部分：直接回归 Sobel 软边界
    l1 = F.l1_loss(pred, edge_gt)

    # 3) soft Dice 部分：关注重叠区域形状
    B = pred.size(0)
    pred_flat = pred.view(B, -1)
    gt_flat   = edge_gt.view(B, -1)

    inter = (pred_flat * gt_flat).sum(dim=1)
    union = (pred_flat * pred_flat).sum(dim=1) + (gt_flat * gt_flat).sum(dim=1)

    dice = (2 * inter + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    loss = alpha * l1 + (1.0 - alpha) * dice_loss
    return loss
