import os
import time
import torch

from wan.configs import WAN_CONFIGS
from wan.text2video import WanT2V
from wan.quant.ptq_minimal import replace_linears_w4, QuantLinearW4A16

def main():
    checkpoint_dir = os.environ.get("WAN_CKPT", "/root/autodl-tmp/Wan2.1-T2V-1.3B")
    qstate_path = os.environ.get("WAN_QSTATE", "wan2.1_w4a16_ptq_minimal.pt")

    cfg = WAN_CONFIGS["t2v-1.3B"]

    # 1) load fp16 pipeline
    pipe = WanT2V(config=cfg, checkpoint_dir=checkpoint_dir, device_id=0)
    model = pipe.model
    # 这里是用来对齐offload_model=True的FP16/量化模型的，如果不需要就删除下面一行
    USE_QUANT = os.environ.get("USE_QUANT", "1") == "1"
    model.eval().requires_grad_(False)

    # 2) IMPORTANT: 先做一次 replace，把结构替换成 QuantLinearW4A16
    # act_stats 这里不再需要真实统计，给空 dict 也行（我们只要完成替换）
  # 这里是用来对齐offload_model=True的FP16/量化模型的，如果不需要就删除下面一段
    if USE_QUANT:
        replace_linears_w4(model, act_stats={})
        sd = torch.load(qstate_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[MODE] Quant enabled (W4A16 minimal)")
        print("[LOAD] missing:", len(missing), "unexpected:", len(unexpected))
    else:
        print("[MODE] FP16 baseline (no quant)")

    # replace_linears_w4(model, act_stats={})

    # # 3) load quantized state (包含 QuantLinearW4A16 的参数名)
    # sd = torch.load(qstate_path, map_location="cpu")
    # missing, unexpected = model.load_state_dict(sd, strict=False)
    # print("[LOAD] missing:", len(missing), "unexpected:", len(unexpected))
    # if len(unexpected) > 0:
    #     print("[LOAD] unexpected sample:", unexpected[:10])
    # if len(missing) > 0:
    #     print("[LOAD] missing sample:", missing[:10])

    # 4) run normal generation (use same settings as your FP16 baseline)
    prompt = "A cat playing guitar on the moon"
    size = (832, 480)
    frame_num = 17  # 你基线是 16
    sampling_steps = 50  # 这里按你 baseline 的 steps 改（你之前显示 Steps:N/A，就先用默认 50）
    seed = 123

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    with torch.no_grad():
        video = pipe.generate(
            input_prompt=prompt,
            size=size,
            frame_num=frame_num,
            sampling_steps=sampling_steps,
            seed=seed,
            offload_model=True,   # 量化对比时建议关 offload，公平
        )

    t1 = time.time()
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[RESULT] Peak VRAM: {peak:.2f} GB, Time: {t1-t0:.2f} s")

    # 保存输出（如果你希望保存 mp4，用官方 generate.py 的保存逻辑更简单）
    # 这里只打印形状：
    if isinstance(video, torch.Tensor):
        print("[VIDEO] tensor shape:", tuple(video.shape))
    else:
        print("[VIDEO] type:", type(video))

if __name__ == "__main__":
    main()
