import torch
import torch.nn as nn

def main():
    from torchao.quantization import quantize_
    # torchao 0.7.0 里 W8A16 config 的名字可能是下面其中一种
    # 我们做一个兼容写法：谁存在用谁
    try:
        from torchao.quantization import Int8WeightOnlyConfig as W8A16Config
        cfg = W8A16Config()
        cfg_name = "Int8WeightOnlyConfig"
    except Exception:
        try:
            from torchao.quantization import int8_weight_only as W8A16Config
            cfg = W8A16Config()
            cfg_name = "int8_weight_only"
        except Exception as e:
            raise RuntimeError("Cannot find W8A16 config in torchao 0.7.0. Please print(dir(torchao.quantization))") from e

    m = nn.Sequential(
        nn.Linear(4096, 4096, bias=False),
        nn.GELU(),
        nn.Linear(4096, 4096, bias=False),
    ).eval().cuda().half()

    # 量化前：权重 dtype
    w0 = m[0].weight
    print("[BEFORE] weight dtype:", w0.dtype, "shape:", tuple(w0.shape))

    # 应用 torchao W8A16
    quantize_(m, cfg)
    print("[APPLIED] cfg:", cfg_name)

    # 量化后：模块类型 + weight 属性形态（不同版本会不同）
    w1 = m[0].weight
    print("[AFTER]  module:", type(m[0]).__name__)
    print("[AFTER]  weight attr type:", type(w1))
    if hasattr(w1, "dtype"):
        print("[AFTER]  weight dtype:", w1.dtype)

    # 跑一次 forward，确保真能算
    x = torch.randn(2, 4096, device="cuda", dtype=torch.float16)
    y = m(x)
    print("[RUN] out:", y.shape, y.dtype)

if __name__ == "__main__":
    main()
