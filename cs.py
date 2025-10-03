import torch
import torch.nn as nn
import argparse
import math
import torch.nn.functional as F
from main import MobileNetV2_CIFAR, InvertedResidual, get_loaders, evaluate, device
import wandb
import pickle

torch.manual_seed(6886)


def find_expansion_convs(model):
    blocks = []
    for name, m in model.named_modules():
        if hasattr(m, "conv") and isinstance(m.conv, nn.Sequential):
            seq = m.conv
            if (
                len(seq) >= 1
                and isinstance(seq[0], nn.Conv2d)
                and tuple(seq[0].kernel_size) == (1, 1)
            ):
                blocks.append((name, m))
    return blocks


def get_prune_masks(model, prune_frac=0.25, verbose=True):
    masks = {}
    blocks = find_expansion_convs(model)
    for name, block in blocks:
        seq = block.conv
        exp_conv = seq[0]
        out_ch = exp_conv.weight.shape[0]
        w = exp_conv.weight.detach().abs().view(out_ch, -1).sum(1)
        k_prune = max(1, int(math.floor(out_ch * prune_frac)))
        prune_idx = torch.topk(w, k_prune, largest=False)[1]
        keep_mask = torch.ones(out_ch, dtype=torch.bool, device=exp_conv.weight.device)
        keep_mask[prune_idx.to(exp_conv.weight.device)] = False
        masks[name] = keep_mask
        if verbose:
            kept = int(keep_mask.sum().item())
            print(f"Prune mask {name}: {out_ch}->{kept}")
    return masks


def make_pruned_block(old_block, input_ch, out_ch, stride, keep_mask, device=device):
    seq_old = old_block.conv
    assert isinstance(seq_old[0], nn.Conv2d) and tuple(seq_old[0].kernel_size) == (1, 1)

    keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1).to(device)
    new_hidden = int(keep_idx.numel())

    conv1 = nn.Conv2d(input_ch, new_hidden, 1, bias=False)
    bn1 = nn.BatchNorm2d(new_hidden)
    relu1 = nn.ReLU6(inplace=True)

    old_dw = seq_old[3]
    dw_k = old_dw.kernel_size
    dw_stride = stride
    depth = nn.Conv2d(
        new_hidden,
        new_hidden,
        dw_k,
        dw_stride,
        padding=old_dw.padding,
        groups=new_hidden,
        bias=False,
    )
    bn2 = nn.BatchNorm2d(new_hidden)
    relu2 = nn.ReLU6(inplace=True)

    proj = nn.Conv2d(new_hidden, out_ch, 1, bias=False)
    bn3 = nn.BatchNorm2d(out_ch)

    new_seq = nn.Sequential(conv1, bn1, relu1, depth, bn2, relu2, proj, bn3)
    new_seq = new_seq.to(device)

    exp_old = seq_old[0]
    new_seq[0].weight.data.copy_(exp_old.weight.data[keep_idx].clone())

    bn1_old = seq_old[1]
    new_seq[1].weight.data.copy_(bn1_old.weight.data[keep_idx].clone())
    new_seq[1].bias.data.copy_(bn1_old.bias.data[keep_idx].clone())
    new_seq[1].running_mean.copy_(bn1_old.running_mean[keep_idx].clone())
    new_seq[1].running_var.copy_(bn1_old.running_var[keep_idx].clone())

    new_seq[3].weight.data.copy_(old_dw.weight.data[keep_idx].clone())

    bn2_old = seq_old[4]
    new_seq[4].weight.data.copy_(bn2_old.weight.data[keep_idx].clone())
    new_seq[4].bias.data.copy_(bn2_old.bias.data[keep_idx].clone())
    new_seq[4].running_mean.copy_(bn2_old.running_mean[keep_idx].clone())
    new_seq[4].running_var.copy_(bn2_old.running_var[keep_idx].clone())

    proj_old = seq_old[6]
    new_seq[6].weight.data.copy_(proj_old.weight.data[:, keep_idx, :, :].clone())

    bn3_old = seq_old[7]
    new_seq[7].weight.data.copy_(bn3_old.weight.data.clone())
    new_seq[7].bias.data.copy_(bn3_old.bias.data.clone())
    new_seq[7].running_mean.copy_(bn3_old.running_mean.clone())
    new_seq[7].running_var.copy_(bn3_old.running_var.clone())

    new_block = InvertedResidual(input_ch, out_ch, stride, expand_ratio=1)
    new_block.conv = new_seq
    new_block.use_residual = old_block.use_residual
    return new_block


def rebuild_pruned_model(base_model, masks):
    cfgs = [
        [1, 16, 1, 1],
        [6, 24, 2, 1],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    new_model = MobileNetV2_CIFAR()
    new_model = new_model.to(device)

    # copy stem unchanged
    new_model.stem.load_state_dict(base_model.stem.state_dict())

    # rebuild features block by block
    new_layers = []
    input_ch = base_model.stem[0].out_channels
    idx = 0
    for t, c, n, s in cfgs:
        out_ch = int(c)
        for i in range(n):
            stride = s if i == 0 else 1
            block = base_model.features[idx]
            name = f"features.{idx}"
            keep_mask = masks.get(
                name, None
            )  # NOTE: key is 'features.X' as produced by get_prune_masks
            if keep_mask is not None:
                # build a reduced-hidden block and copy weights
                new_block = make_pruned_block(
                    block, input_ch, out_ch, stride, keep_mask, device=device
                )
            else:
                # no pruning for this block -> copy original block directly
                new_block = InvertedResidual(input_ch, out_ch, stride, t)
                new_block.load_state_dict(block.state_dict())
                new_block = new_block.to(device)
            new_layers.append(new_block)
            input_ch = out_ch  # block output channels remain the same (we pruned expansion, not proj)
            idx += 1

    new_model.features = nn.Sequential(*new_layers)

    # copy head: head.conv expects input_ch == last block out_ch, which we preserved,
    # so we can copy weights directly (shapes should match)
    new_model.head.load_state_dict(base_model.head.state_dict())
    new_model.classifier.load_state_dict(base_model.classifier.state_dict())

    return new_model


def finetune(model, train_loader, val_loader, epochs=10, lr=0.02):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        corr = tot = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            _, p = out.max(1)
            corr += p.eq(y).sum().item()
            tot += y.size(0)
        sched.step()
        vloss, vacc = evaluate(model, val_loader, crit)
        print(
            f"FT {ep + 1}/{epochs}: train_acc={corr / tot * 100:.2f}% val_acc={vacc * 100:.2f}% loss={vloss:.4f}"
        )
    return model


def fake_quant(x, bits=8, signed=False):
    if signed:
        max_abs = x.abs().max().clamp_min(1e-12)
        qmax = (1 << (bits - 1)) - 1
        scale = max_abs / float(qmax)
        q = torch.round(x / scale).clamp(-qmax, qmax)
        dq = q * scale
    else:
        mn, mx = x.min(), x.max()
        qmin, qmax = 0, (1 << bits) - 1
        rng = float(mx.detach()) - float(mn.detach())
        if rng < 1e-12:
            return x.detach()
        scale = rng / float(qmax - qmin)
        zp = round(-float(mn) / scale)
        q = torch.round(x / scale + zp).clamp(qmin, qmax)
        dq = (q - zp) * scale
    return (dq - x).detach() + x


class QuantWrapper(nn.Module):
    def __init__(self, m, w_bits=8, a_bits=8, per_channel=True):
        super().__init__()
        self.m = m
        self.wb = int(w_bits)
        self.ab = int(a_bits)
        self.pc = per_channel

    def forward(self, x):
        W = self.m.weight
        if isinstance(self.m, nn.Conv2d) and self.pc:
            flat = W.view(W.size(0), -1)
            max_abs = flat.abs().max(1)[0].clamp_min(1e-12)
            qmax = (1 << (self.wb - 1)) - 1
            scale = max_abs / qmax
            q = torch.round(flat / scale[:, None]).clamp(-qmax, qmax)
            dq = (q * scale[:, None]).view_as(W)
        else:
            dq = fake_quant(W, self.wb, signed=True)
        if isinstance(self.m, nn.Conv2d):
            out = F.conv2d(
                x,
                dq,
                self.m.bias,
                stride=self.m.stride,
                padding=self.m.padding,
                dilation=self.m.dilation,
                groups=self.m.groups,
            )
        else:
            out = F.linear(x, dq, self.m.bias)
        return fake_quant(out, self.ab, signed=False)


def attach_qwrappers(model, w_bits, a_bits):
    for name, mod in model.named_children():
        if isinstance(mod, nn.Conv2d):
            if mod.in_channels == 3:
                continue
            setattr(model, name, QuantWrapper(mod, w_bits, a_bits))
        elif isinstance(mod, nn.Linear):
            continue
        else:
            attach_qwrappers(mod, w_bits, a_bits)


def qat_finetune(model, train_loader, val_loader, epochs=10, lr=0.01):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        corr = tot = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            _, p = out.max(1)
            corr += p.eq(y).sum().item()
            tot += y.size(0)
        sched.step()
        vloss, vacc = evaluate(model, val_loader, crit)
        print(
            f"QAT {ep + 1}/{epochs}: train_acc={corr / tot * 100:.2f}% val_acc={vacc * 100:.2f}% loss={vloss:.4f}"
        )
    return model


def measure_activation_counts(model, loader, device, batches=1):
    total_elems = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= batches:
                break
            x = x.to(device)
            elems = 0

            def hook_fn(_, __, out):
                nonlocal elems
                if isinstance(out, torch.Tensor):
                    elems += out.numel()
                elif isinstance(out, (list, tuple)):
                    elems += sum(o.numel() for o in out if isinstance(o, torch.Tensor))

            hooks = [
                m.register_forward_hook(hook_fn)
                for m in model.modules()
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6))
            ]
            _ = model(x)
            for h in hooks:
                h.remove()
            total_elems += elems
            n += 1
    return total_elems // max(1, n)


def log_model_size(base_params, pruned_params, w_bits, a_bits, act_elements, q_acc):
    base_bytes = base_params * 4
    quant_bytes = int(pruned_params * w_bits / 8)
    ratio = base_bytes / max(1, quant_bytes)
    print(
        f"[SIZE] BaseFP32={base_bytes / 1024 / 1024:.3f}MB "
        f"Pruned+Quant({w_bits}-bit)≈{quant_bytes / 1024 / 1024:.3f}MB "
        f"CompRatio={ratio:.2f}x "
        f"(params {pruned_params} vs base {base_params})"
    )
    base_act_bytes = act_elements * 4
    quant_act_bytes = int(act_elements * a_bits / 8)
    act_ratio = base_act_bytes / max(1, quant_act_bytes)
    print(
        f"[SIZE] Activation footprint (per fwd est): "
        f"FP32={base_act_bytes / 1024 / 1024:.3f}MB "
        f"Quant({a_bits}-bit)≈{quant_act_bytes / 1024 / 1024:.3f}MB "
        f"CompRatio(acts)={act_ratio:.2f}x"
    )
    wandb.log(
        {
            "w_bits": w_bits,
            "a_bits": a_bits,
            "compression_ratio": ratio,
            "model_size_mb": quant_bytes / 1024 / 1024,
            "final_accuracy": q_acc * 100,
            "activation_compression_ratio": act_ratio,
            "activation_size_mb": quant_act_bytes / 1024 / 1024,
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune_frac", type=float, default=0.25)
    ap.add_argument("--ft_epochs", type=int, default=15)
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--a_bits", type=int, default=8)
    ap.add_argument("--qat_epochs", type=int, default=10)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    config_identifier = f"pf{args.prune_frac}_ft{args.ft_epochs}_wb{args.w_bits}_ab{args.a_bits}_qat{args.qat_epochs}"
    wandb.init(name=config_identifier)

    train, val, test = get_loaders(batch_size=256)

    model = MobileNetV2_CIFAR().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    base_params_count = sum(p.numel() for p in model.parameters())

    crit = nn.CrossEntropyLoss()
    base_loss, base_acc = evaluate(model, test, crit)
    print(f"[BASE] acc={base_acc * 100:.2f}% loss={base_loss:.4f}")

    masks = get_prune_masks(model, args.prune_frac, verbose=True)
    model = rebuild_pruned_model(model, masks)
    print("Pruned param count:", sum(p.numel() for p in model.parameters()))
    pre_loss, pre_acc = evaluate(model, test, crit)
    print(f"[PRUNED BEFORE FT] acc={pre_acc * 100:.2f}% loss={pre_loss:.4f}")

    if args.ft_epochs > 0:
        model = finetune(model, train, val, epochs=args.ft_epochs)
    post_loss, post_acc = evaluate(model, test, crit)
    print(f"[PRUNED AFTER FT] acc={post_acc * 100:.2f}% loss={post_loss:.4f}")

    print(f"Attaching QWrappers: w_bits={args.w_bits}, a_bits={args.a_bits}")
    attach_qwrappers(model, args.w_bits, args.a_bits)

    if args.qat_epochs > 0:
        model = qat_finetune(model, train, val, epochs=args.qat_epochs)
    q_loss, q_acc = evaluate(model, test, crit)
    print(f"[QAT FINAL] acc={q_acc * 100:.2f}% loss={q_loss:.4f}")

    act_counts = measure_activation_counts(model, test, device, batches=1)
    pruned_params = sum(p.numel() for p in model.parameters())
    log_model_size(
        base_params_count, pruned_params, args.w_bits, args.a_bits, act_counts, q_acc
    )

    torch.save(model.state_dict(), "compressed_best.pth")
    with open("masks.pkl", "wb") as f:
        pickle.dump(masks, f)

    wandb.finish()


if __name__ == "__main__":
    main()
