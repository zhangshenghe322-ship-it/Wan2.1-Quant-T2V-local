# tools/calibrate_w4a16.py
import os
import torch
from easydict import EasyDict

from wan.text2video import WanT2V
from wan.quant.ptq_minimal import register_act_hooks, replace_linears_w4, save_quantized_state

def main():
    print("[PTQ] Enter calibrate_w4a16 main()")

    # 你这里按你项目的 config 加载方式改一下
    # 假设你有 config 对象和 checkpoint_dir
    from wan.configs import WAN_CONFIGS

    # 选择与你 checkpoint 对应的配置名（1.3B T2V）
    cfg = WAN_CONFIGS["t2v-1.3B"]

    checkpoint_dir = os.environ.get("CKPT_DIR", "/root/autodl-tmp/Wan2.1-T2V-1.3B")

    pipe = WanT2V(config=cfg, checkpoint_dir=checkpoint_dir, device_id=0)
    print("[PTQ] WanT2V initialized")

    model = pipe.model
    model.eval().requires_grad_(False)

    # ===== 1) 构造一个最小可跑的 calibration 输入 =====
    # 用模型真实输入通道，避免 patch_embedding 通道不匹配
    C_in = model.patch_embedding.in_channels  # ✅ 直接读模型
    F = 5   # 4n+1，最小视频长度
    H = 32
    W = 32

    # x: List[Tensor]，每个 [C_in, F, H, W]
    x = [torch.randn(C_in, F, H, W, device=model.patch_embedding.weight.device, dtype=torch.float32)]

    # t: timesteps，shape [B]
    t = torch.zeros(1, device=model.patch_embedding.weight.device, dtype=torch.long)

    # context: 用 text_encoder 正规拿（返回就是 List[Tensor]）
    prompt = "A cat playing guitar on the moon"
    with torch.no_grad():
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)  # List[Tensor], each [L, C]
        pipe.text_encoder.model.cpu()  # 省显存可选

    # seq_len: 给个足够大的安全值，避免 assert
    seq_len = 8192

    # ===== 2) dtype/device 对齐：以 patch_embedding 为准 =====
    pe_dtype = model.patch_embedding.weight.dtype
    pe_device = model.patch_embedding.weight.device
    print("[PTQ] patch_embedding dtype:", pe_dtype, "device:", pe_device)

    x = [u.to(device=pe_device, dtype=pe_dtype) for u in x]
    context = [u.to(device=pe_device, dtype=pe_dtype) for u in context]
    t = t.to(device=pe_device)  # t 保持 long

    # ===== 3) 注册 hooks + forward 校准 =====
    act_stats = {}
    hooks = register_act_hooks(model, act_stats)

    print("[PTQ] Running calibration forward...")
    with torch.no_grad():
        _ = model(x=x, t=t, context=context, seq_len=seq_len)

    for h in hooks:
        h.remove()

    print(f"[PTQ] Collected act stats for {len(act_stats)} linear layers")

    # ===== 4) 替换 Linear -> QuantLinear，并保存 =====
    replace_linears_w4(model, act_stats)
    out_path = "wan2.1_w4a16_ptq_minimal.pt"
    save_quantized_state(model, out_path)
    print("[PTQ] Saved quantized state to:", out_path)


if __name__ == "__main__":
    main()
