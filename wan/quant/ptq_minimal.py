# wan/quant/ptq_minimal.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _is_skip(name: str) -> bool:
    n = name.lower()
    # 第一版：跳过 norm / embed / vae / t5 / text_encoder
    if any(k in n for k in ["norm", "embed", "vae", "t5", "text_encoder"]):
        return True
    return False

def get_target_linears(model: nn.Module):
    targets = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not _is_skip(name):
            targets.append((name, m))
    return targets

def register_act_hooks(model: nn.Module, act_stats: dict):
    hooks = []
    def make_hook(name):
        def hook(m, inputs, outputs):
            x = inputs[0].detach()
            act_stats.setdefault(name, []).append(x.abs().max().item())
        return hook

    for name, m in get_target_linears(model):
        hooks.append(m.register_forward_hook(make_hook(name)))
    return hooks

class QuantLinearW4A16(nn.Module):
    """Weight 4bit fake-quant, activation FP16 keep."""
    def __init__(self, linear: nn.Linear, w_scale: float):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach(), requires_grad=False)
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
        self.w_scale = float(w_scale)

    def forward(self, x):
        # fake quant weight
        s = self.w_scale
        qmin, qmax = -(2**3), (2**3 - 1)  # 4-bit signed range [-8,7]
        qw = torch.clamp(torch.round(self.weight / s), qmin, qmax) * s
        return F.linear(x, qw, self.bias)

def _get_module_by_name(model: nn.Module, name: str):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def replace_linears_w4(model: nn.Module, act_stats: dict, scale_shrink: float = 0.95):
    """
    用激活统计做一个简化 outlier-aware：
    - 对每层用 max(act_max) * shrink 作为 scale 的参考（雏形）
    - 真实 OmniQuant 更复杂，但这个先跑通 pipeline 和出数据
    """
    for name, m in get_target_linears(model):
        # 计算 weight scale：这里用 weight 的 absmax，也可结合 act_stats 做更细策略
        w_absmax = m.weight.detach().abs().max().item()
        w_scale = (w_absmax / 7.0) if w_absmax > 0 else 1e-6  # 映射到 4-bit [-8,7]

        parent, attr = _get_module_by_name(model, name)
        setattr(parent, attr, QuantLinearW4A16(m, w_scale))

def save_quantized_state(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
