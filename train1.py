import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import losses as loss_utils
from utils import test_util
from dataloaders.dataset import *
from networks.DWT_edge import VNet
import random
import numpy as np
import torch
import torch.nn.functional as F
import json
import re
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BTCV')
parser.add_argument('--root_path', type=str, default='../data1/btcv_h5')
parser.add_argument('--model', type=str, default='edge_cluster4DWT')
parser.add_argument('--max_iteration', type=int, default=8000)
parser.add_argument('--total_samples', type=int, default=30)
parser.add_argument('--max_train_samples', type=int, default=18)
parser.add_argument('--max_val_samples',   type=int, default=6)
parser.add_argument('--max_test_samples',  type=int, default=6)
parser.add_argument('--save_checkpoint_freq', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--val_every_epochs', type=int, default=20,
                    help='run validation every N epochs (default: 1 means every epoch)')
parser.add_argument('--early_stop_patience', type=int, default=180,
                    help='early stopping patience in epochs (counts epochs, not validations)')
parser.add_argument('--min_delta', type=float, default=0.0005,
                    help='minimum val loss decrease to reset patience')
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--edge_loss_weight', type=float, default=0.05,
                    help='weight λ for edge loss in total loss: L = L_seg + λ * L_edge')

args = parser.parse_args()


def create_model(n_classes=14, patchsize=96, ema=False):
    net = VNet(n_channels=1, n_classes=n_classes, patch_size=patchsize)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for p in model.parameters():
            p.detach_()
    return model


snapshot_path = "../model/{}_{}".format(args.dataset_name, args.model)
txt_path = "../data/btcv.txt"

num_classes = 14
patch_size = (96, 96, 96)
train_data_path = args.root_path
max_iterations = args.max_iteration
base_lr = args.base_lr


# ---------------- 软边界损失（Sobel edge 监督） ----------------
def compute_soft_edge_loss(edge_logits: torch.Tensor,
                           edge_gt: torch.Tensor,
                           alpha: float = 0.5,
                           smooth: float = 1e-6) -> torch.Tensor:
    """
    针对预处理阶段用 Sobel 提取的软边界 (0~1) 的边界损失：
        L_edge = alpha * L1 + (1-alpha) * Dice

    Args:
        edge_logits: [B,1,D,H,W]  (网络预测的边界 logits)
        edge_gt    : [B,1,D,H,W] 或 [B,D,H,W] (Sobel 边界, 0~1)
        alpha      : L1 与 Dice 权重
        smooth     : Dice 平滑项

    Returns:
        标量 loss (tensor)
    """
    # 对齐形状
    if edge_gt.dim() == 4:
        edge_gt = edge_gt.unsqueeze(1)
    edge_gt = edge_gt.float()

    if edge_logits.shape != edge_gt.shape:
        raise ValueError(f"shape mismatch: edge_logits {edge_logits.shape} vs edge_gt {edge_gt.shape}")

    # 1) 概率
    pred = torch.sigmoid(edge_logits)  # [B,1,...]

    # 2) L1
    l1 = F.l1_loss(pred, edge_gt)

    # 3) soft Dice
    B = pred.size(0)
    pred_flat = pred.view(B, -1)
    gt_flat = edge_gt.view(B, -1)

    inter = (pred_flat * gt_flat).sum(dim=1)
    union = (pred_flat * pred_flat).sum(dim=1) + (gt_flat * gt_flat).sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    loss = alpha * l1 + (1.0 - alpha) * dice_loss
    return loss


# 日志
def config_log(snapshot_path_tmp, typename):
    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join(snapshot_path_tmp, f"log_{typename}.txt"), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def make_val_transform(patch_size):
    """验证集固定裁剪：PadIfNeeded + 中心裁剪 + ToTensor3D"""
    return transforms.Compose([
        PadIfNeeded3D(patch_size),
        CenterCrop3D(patch_size),
        ToTensor3D(),
    ])


def validate_loss(model, valloader, seg_criterion, edge_weight, global_iter, writer=None):
    """
    验证阶段计算组合损失:
        L_total = L_seg + λ * L_edge
    用于 early stopping.
    """
    model.eval()
    running_total, running_seg, running_edge, n_samples = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in valloader:
            vol = batch['image'].cuda()
            seg = batch['label'].cuda()
            edge_gt = batch.get('edge', None)
            if edge_gt is not None:
                edge_gt = edge_gt.cuda()

            out = model(vol)
            # VNet: (out_seg, embedding, edge_logits)
            if isinstance(out, (tuple, list)):
                out_seg, _, edge_logits = out
            else:
                # 兼容性：如果未来模型只返回 seg，则不计算边界损失
                out_seg = out
                edge_logits = None

            seg_loss = seg_criterion(out_seg, seg)
            if edge_gt is not None and edge_logits is not None:
                edge_loss = compute_soft_edge_loss(edge_logits, edge_gt)
            else:
                edge_loss = seg_loss.detach() * 0.0

            total_loss = seg_loss + edge_weight * edge_loss

            bs = vol.size(0)
            running_total += total_loss.item() * bs
            running_seg += seg_loss.item() * bs
            running_edge += edge_loss.item() * bs
            n_samples += bs

    avg_total = running_total / max(1, n_samples)
    avg_seg = running_seg / max(1, n_samples)
    avg_edge = running_edge / max(1, n_samples)

    if writer is not None:
        writer.add_scalar('Val/Loss_total', avg_total, global_iter)
        writer.add_scalar('Val/Loss_seg', avg_seg, global_iter)
        writer.add_scalar('Val/Loss_edge', avg_edge, global_iter)

    return avg_total


def validate_full_metrics(model, image_id_list, global_iter, writer=None, tag_prefix='Test'):
    """测试阶段：完整滑窗评估（不使用边界损失，只看分割 Dice）"""
    model.eval()
    with torch.no_grad():
        dice_all, std_all, metric_all_cases = test_util.validation_all_case(
            model, num_classes=num_classes, base_dir=train_data_path,
            image_list=image_id_list, patch_size=patch_size,
            stride_xy=32, stride_z=32
        )
    dice_avg = float(dice_all.mean())
    if writer is not None:
        writer.add_scalar(f'{tag_prefix}/Dice_Avg', dice_avg, global_iter)
        organ_names = [
            'spleen', 'r.kidney', 'l.kidney', 'gallbladder', 'esophagus',
            'liver', 'stomach', 'aorta', 'inferior vena cava',
            'portal vein and splenic vein', 'pancreas',
            'right adrenal gland', 'left adrenal gland'
        ]
        for i, organ in enumerate(organ_names):
            writer.add_scalar(f'{tag_prefix}/Dice_{organ}', dice_all[i], global_iter)
    return dice_avg, dice_all, std_all, metric_all_cases


def train(train_ids, val_ids, test_ids, fold_id=1):
    snapshot_path_tmp = snapshot_path
    os.makedirs(snapshot_path_tmp, exist_ok=True)
    handler, sh = config_log(snapshot_path_tmp, 'fold' + str(fold_id))
    logging.info(str(args))

    # 随机性
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    model = create_model(n_classes=num_classes, patchsize=patch_size[0])
    start_iter = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading model: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cuda')
            model.load_state_dict(checkpoint)
            try:
                m = re.search(r'iter_(\d+)', os.path.basename(args.resume))
                start_iter = int(m.group(1)) if m else 0
            except Exception:
                start_iter = 0
        else:
            logging.error(f"Checkpoint not found: {args.resume}")
            sys.exit(1)

    # ------- Dataloaders -------
    train_tf = transforms.Compose([
        RandomTranslate3D(),
        RandomCrop3D(patch_size),
        RandomMixedNoise3D(
            std_range=(0.01, 0.06),
            p=0.3,
            noise_type="mixed",
            salt_pepper_ratio=0.0005
        ),
        ToTensor3D(),
    ])

    db_train = BTCV(train_ids, base_dir=train_data_path, transform=train_tf)
    trainloader = DataLoader(
        db_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(args.seed + x)
    )

    val_tf = make_val_transform(patch_size)  # PadIfNeeded3D + CenterCrop3D + ToTensor3D
    db_val = BTCV(val_ids, base_dir=train_data_path, transform=val_tf)
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=1e-4
    )

    writer = SummaryWriter(snapshot_path_tmp)
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # 主分割损失（Dice）
    seg_criterion = loss_utils.DiceLoss(n_classes=num_classes)
    edge_weight = args.edge_loss_weight

    iter_num = start_iter
    metric_all_cases_test_final = None

    # 早停相关（按 epoch 计数）
    best_val = float('inf')
    epochs_since_improve = 0

    max_epoch = (max_iterations - start_iter) // max(1, len(trainloader)) + 2
    iterator = tqdm(range(max_epoch), ncols=70)

    plot_losses = []

    for epoch_num in iterator:
        # -------- Train one epoch --------
        for i_batch, batch in enumerate(trainloader):
            if iter_num >= max_iterations:
                break

            vol = batch['image'].cuda()
            seg = batch['label'].cuda()
            edge_gt = batch.get('edge', None)
            if edge_gt is not None:
                edge_gt = edge_gt.cuda()

            model.train()

            # ====== poly lr ======
            cur_lr = base_lr * (1.0 - float(iter_num) / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr
            writer.add_scalar('LearningRate', cur_lr, iter_num)

            # ====== 前向 ======
            out = model(vol)
            # VNet: (out_seg, embedding, edge_logits)
            if isinstance(out, (tuple, list)):
                out_seg, _, edge_logits = out
            else:
                out_seg = out
                edge_logits = None

            # ====== 损失 ======
            seg_loss = seg_criterion(out_seg, seg)
            if edge_gt is not None and edge_logits is not None:
                edge_loss = compute_soft_edge_loss(edge_logits, edge_gt)
            else:
                edge_loss = seg_loss.detach() * 0.0

            total_loss = seg_loss + edge_weight * edge_loss

            writer.add_scalar('Loss/train_total', total_loss.item(), iter_num)
            writer.add_scalar('Loss/train_seg', seg_loss.item(), iter_num)
            writer.add_scalar('Loss/train_edge', edge_loss.item(), iter_num)

            # ====== 反向 / 更新 ======
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iter_num += 1

            if iter_num % 60 == 0:
                plot_losses.append((iter_num, total_loss.item()))
                logging.info(
                    f'Fold {fold_id}, epoch {epoch_num}, iter {iter_num}: '
                    f'total_loss {total_loss.item():.3f}, seg_loss {seg_loss.item():.3f}, '
                    f'edge_loss {edge_loss.item():.3f}, lr {cur_lr:.6f}'
                )

            if args.save_checkpoint_freq > 0 and iter_num % args.save_checkpoint_freq == 0:
                ckpt = os.path.join(snapshot_path_tmp, f'iter_{iter_num}_checkpoint.pth')
                torch.save(model.state_dict(), ckpt)
                logging.info(f"Saved checkpoint: {ckpt}")

        # -------- Validation schedule（每 N 个 epoch 验证一次）-------- 
        improved_this_epoch = False
        if (epoch_num + 1) % args.val_every_epochs == 0:
            val_loss = validate_loss(model, valloader, seg_criterion, edge_weight, iter_num, writer)
            logging.info(f'Epoch {epoch_num}, iter {iter_num}, Val total loss: {val_loss:.6f}')

            if val_loss < (best_val - args.min_delta):
                best_val = val_loss
                improved_this_epoch = True
                epochs_since_improve = 0

                # 保存 best 模型（基于总验证损失）
                best_path = os.path.join(snapshot_path_tmp, f'{args.model}_best_model.pth')
                torch.save(model.state_dict(), best_path)
                logging.info(f'Improved! Saved BEST model (by ValTotalLoss) to {best_path}')

        # -------- 早停计数（按 epoch）--------
        if not improved_this_epoch:
            epochs_since_improve += 1

        # 判定早停
        if epochs_since_improve >= args.early_stop_patience:
            logging.info(
                f'Early stopping triggered at epoch {epoch_num} '
                f'(no ValLoss improvement for {epochs_since_improve} epochs).'
            )
            break

        if iter_num >= max_iterations:
            logging.info('Reached max_iterations. Stop training loop.')
            break

    # -------- Save FINAL model (always) --------
    final_path = os.path.join(snapshot_path_tmp, f'{args.model}_final.pth')
    torch.save(model.state_dict(), final_path)
    logging.info(f"Saved FINAL model to {final_path}")

    # -------- Choose model for Test: prefer BEST if exists --------
    best_path = os.path.join(snapshot_path_tmp, f'{args.model}_best_model.pth')
    model_to_test = best_path if os.path.exists(best_path) else final_path
    model.load_state_dict(torch.load(model_to_test, map_location='cuda'))
    logging.info(f'Loaded model for test: {model_to_test}')

    # -------- Final Test (once) --------
    test_dice_avg, test_dice_all, test_std_all, metric_all_cases_test = validate_full_metrics(
        model, test_ids, iter_num, writer, tag_prefix='Test'
    )
    metric_all_cases_test_final = metric_all_cases_test

    logging.info(f'Final Test average DSC: {test_dice_avg:.4f}')
    organ_names = [
        'spleen', 'r.kidney', 'l.kidney', 'gallbladder', 'esophagus',
        'liver', 'stomach', 'aorta', 'inferior vena cava',
        'portal vein and splenic vein', 'pancreas',
        'right adrenal gland', 'left adrenal gland'
    ]
    for i, organ in enumerate(organ_names):
        logging.info(f'  [Test] {organ}: {test_dice_all[i]:.3f}')

    # 可视化曲线（总损失）
    if plot_losses:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        iters, losses = zip(*plot_losses)
        plt.figure(figsize=(10, 6))
        plt.plot(iters, losses, linewidth=1.2)
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss (Seg + Edge)')
        plt.title('Training Total Loss Curve')
        plt.grid(True, linestyle='--', alpha=0.6)
        fig_path = os.path.join(snapshot_path_tmp, 'training_loss.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        np.save(os.path.join(snapshot_path_tmp, 'training_loss.npy'), np.array(plot_losses))
        logging.info(f"Training loss curve saved: {fig_path}")

    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
    return metric_all_cases_test_final


if __name__ == "__main__":

    os.makedirs(snapshot_path, exist_ok=True)
    code_dir = os.path.join(snapshot_path, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    shutil.copytree('.', code_dir, ignore=shutil.ignore_patterns('.git', '__pycache__'))

    # 读全量列表（txt 内是文件名；取末4位为 ID）
    with open(txt_path, 'r') as f:
        raw_lines = [line.strip() for line in f.readlines()]

    total_ids = [item.split('.')[0][-4:] for item in raw_lines]
    total_n = len(total_ids)
    assert args.total_samples <= total_n, f"total_samples({args.total_samples}) > available({total_n})"
    use_ids = total_ids[:args.total_samples]

    # 三划分（按数量顺序切分；训练 ID 再随机打乱）
    mt, mv, ms = args.max_train_samples, args.max_val_samples, args.max_test_samples
    assert mt + mv + ms <= len(use_ids), "train/val/test samples exceed total_samples"

    train_ids = use_ids[:mt]
    val_ids   = use_ids[mt:mt+mv]
    test_ids  = use_ids[mt+mv:mt+mv+ms]
    np.random.shuffle(train_ids)

    # 存分割
    with open(os.path.join(snapshot_path, 'train_ids.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(snapshot_path, 'val_ids.txt'), 'w') as f:
        f.write('\n'.join(val_ids))
    with open(os.path.join(snapshot_path, 'test_ids.txt'), 'w') as f:
        f.write('\n'.join(test_ids))

    # 训练
    metric_final = train(train_ids, val_ids, test_ids)

    # 保存最终测试统计（与原脚本兼容）
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)
    organ_names = [
        'spleen', 'r.kidney', 'l.kidney', 'gallbladder', 'esophagus',
        'liver', 'stomach', 'aorta', 'inferior vena cava',
        'portal vein and splenic vein', 'pancreas',
        'right adrenal gland', 'left adrenal gland'
    ]

    try:
        dice_mean = metric_mean[0]
        dice_std = metric_std[0]
        dice_txt = os.path.join(snapshot_path, 'organ_dice_scores.txt')
        with open(dice_txt, 'w') as f:
            f.write("Organ-wise Dice on Test Set\n" + "="*40 + "\n")
            for i, organ in enumerate(organ_names):
                f.write(f"{organ}: {dice_mean[i]:.3f} ± {dice_std[i]:.3f}\n")
        dice_json = os.path.join(snapshot_path, 'organ_dice_scores.json')
        dice_dict = {organ: {"mean": round(float(dice_mean[i]), 3),
                             "std": round(float(dice_std[i]), 3)}
                     for i, organ in enumerate(organ_names)}
        with open(dice_json, 'w') as f:
            json.dump(dice_dict, f, indent=4)
        logging.info(f"Saved organ-wise DSC to {dice_txt} and {dice_json}")
    except Exception as e:
        logging.error(f"Failed to save organ-wise DSC: {e}")

    exp_name = getattr(args, 'exp', 'default')
    np.save(os.path.join(snapshot_path, f'metric_final_{args.dataset_name}_{exp_name}.npy'), metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info(
        'Final Test (Avg over cases) — '
        'Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}'.format(
            metric_mean[0].mean(), metric_mean[1].mean(),
            metric_mean[2].mean(), metric_mean[3].mean()
        )
    )
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
