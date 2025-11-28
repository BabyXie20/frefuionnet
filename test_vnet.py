import time
import torch
from torch import nn
from networks.cluster_DWT import VNet

def add_flops_counter(model):
    """
    给模型所有 Conv3d / ConvTranspose3d 注册 forward hook，
    统计一次 forward 过程中的 FLOPs 数（只算卷积，其他层忽略）。
    """
    flops_dict = {'total': 0}
    handles = []

    def conv_hook(module, inputs, output):
        x = inputs[0]
        # x: [B, Cin, D_in, H_in, W_in]
        # output: [B, Cout, D_out, H_out, W_out]
        B = x.shape[0]
        out = output
        Cout = out.shape[1]
        D_out, H_out, W_out = out.shape[2:]

        # weight: [Cout, Cin/groups, kD, kH, kW]
        # 每个输出元素的运算：Cin/groups * kD*kH*kW 次乘加
        # 使用 weight[0] 的 numel() 作为 (Cin/groups * kD*kH*kW)
        kernel_ops = module.weight[0].numel() * 2  # 乘+加 → ×2
        conv_flops = B * Cout * D_out * H_out * W_out * kernel_ops
        flops_dict['total'] += conv_flops

    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            h = m.register_forward_hook(conv_hook)
            handles.append(h)

    return flops_dict, handles


def remove_flops_counter(handles):
    for h in handles:
        h.remove()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # 配置
    B, C, D, H, W = 6, 1, 96, 96, 96
    num_classes = 14
    n_filters = 16

    # 建模
    model = VNet(
        n_channels=C, n_classes=num_classes, patch_size=96,
        n_filters=n_filters, has_dropout=False, has_residual=False
    ).to(device)
    model.train()

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e6:.2f}M")

    # 假数据
    x = torch.randn(B, C, D, H, W, device=device)
    y = torch.randint(0, num_classes, (B, D, H, W), device=device, dtype=torch.long)

    # ================
    #  FLOPs 统计部分
    # ================
    flops_dict, handles = add_flops_counter(model)

    # 前向（带 FLOPs 统计）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    logits, embedding = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    # 计算 FLOPs
    total_flops = flops_dict['total']          # 单次 forward 的总 FLOPs
    gflops = total_flops / 1e9
    tflops = total_flops / 1e12
    latency = t1 - t0
    tflops_per_sec = tflops / latency if latency > 0 else float('inf')

    print(f"\n===== FLOPs 统计（单次 forward）=====")
    print(f"FLOPs: {gflops:.3f} GFLOPs  ({tflops:.3f} TFLOPs)")
    print(f"Time : {latency*1000:.2f} ms  → 约 {tflops_per_sec:.2f} TFLOPs/s (仅卷积理论算力)\n")

    # 用完记得把 hook 去掉（防止后续重复统计）
    remove_flops_counter(handles)

    # 原来的 shape 检查
    print("logits   :", tuple(logits.shape))
    print("embed    :", tuple(embedding.shape))

    assert logits.shape == (B, num_classes, D, H, W), "logits 形状不符合预期"
    assert embedding.shape == (B, n_filters, D, H, W), "embedding 形状不符合预期"

    # 单步训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print(f"train step ok. loss={loss.item():.4f}")

    # 推理：softmax + argmax
    model.eval()
    with torch.no_grad():
        logits_eval, _ = model(x)
        probs = torch.softmax(logits_eval, dim=1)  # (B, C, D, H, W)
        pred  = probs.argmax(dim=1)                # (B, D, H, W)
        print("pred shape:", tuple(pred.shape))
        print("pred unique labels (sample):",
              [t.tolist() for t in torch.unique(pred, sorted=True, return_counts=False).unsqueeze(0)[:1]])


if __name__ == "__main__":
    main()
