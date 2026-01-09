Wan2.1-T2V Quantization Evaluation

Reproducible FP16 vs TorchAO W8A16 (INT8 weight-only) evaluation pipeline for Wan2.1-T2V-1.3B on consumer GPUs.

本项目提供 Wan2.1-T2V-1.3B 文本到视频扩散模型在 FP16 与 W8A16（INT8 权重量化） 条件下的完整复现流程，包括视频生成、量化推理以及三项客观视频质量评测指标。

✨ Features

🔹 True INT8 weight-only quantization via TorchAO

🔹 FP16 vs W8A16 paired video generation

🔹 Three objective video metrics:

CLIP alignment

Temporal consistency

Motion magnitude (optical flow)

🔹 Strict prompt/seed pairing for fair comparison

🔹 Two-environment design (generation / evaluation)

🔹 Paper-ready reproducibility workflow

📁 Project Structure
Wan2.1-main/
│
├── dataset/
│   ├── fp16/
│   └── w8a16/
│
├── eval/
│   ├── eval_motion.py
│   └── out/
│
├── eval_scripts/
│   └── eval_clip_temporal_simple.py
│
├── generate.py
├── run_exp.py
└── README.md

🔧 Environments

本项目使用两个 Conda 环境：

Environment	Usage
Wan1	视频生成 + 量化推理
clip_eval	视频质量评测
🚀 Quick Start
1️⃣ Activate Wan1 (Generation)
conda activate Wan1
cd /root/autodl-tmp/Wan2.1-main

2️⃣ Generate Videos
FP16
USE_QUANT=0 python generate.py \
  --task t2v-1.3B \
  --ckpt_dir /root/autodl-tmp/Wan2.1-T2V-1.3B \
  --prompt "A cat playing guitar on the moon" \
  --size 832*480 \
  --frame_num 16 \
  --sample_steps 20 \
  --base_seed 123 \
  --offload_model True \
  --save_file dataset/fp16/01_seed123.mp4

W8A16
USE_QUANT=1 python generate.py \
  --task t2v-1.3B \
  --ckpt_dir /root/autodl-tmp/Wan2.1-T2V-1.3B \
  --prompt "A cat playing guitar on the moon" \
  --size 832*480 \
  --frame_num 16 \
  --sample_steps 20 \
  --base_seed 123 \
  --offload_model True \
  --save_file dataset/w8a16/01_seed123.mp4


Repeat for 20 prompts with identical seeds.

3️⃣ Activate clip_eval (Evaluation)
conda activate clip_eval
cd /root/autodl-tmp/Wan2.1-main


Install dependencies:

pip install torch torchvision transformers pillow numpy pandas tqdm
pip install opencv-python imageio imageio-ffmpeg

📊 Evaluation

Create output directory:

mkdir -p eval/out

CLIP Alignment + Temporal Consistency
python eval_scripts/eval_clip_temporal_simple.py \
  --fp16_dir dataset/fp16 \
  --w8a16_dir dataset/w8a16 \
  --out_dir eval/out


Outputs:

clip_temporal_per_video.csv

clip_temporal_summary.csv

Motion Magnitude (Optical Flow)
python eval/eval_motion.py \
  --fp16_dir dataset/fp16 \
  --w8a16_dir dataset/w8a16 \
  --out_dir eval/out


Output:

motion_summary.csv

📈 Metrics Reported
Metric	Description
CLIP Score	Text-video semantic alignment
Temporal Consistency	Frame-level stability
Motion Magnitude	Optical flow strength

All metrics are computed on paired videos under identical inference settings.

🔍 Quantization Details

Backend: TorchAO

Mode: INT8 weight-only (W8A16)

Quantized: Transformer Linear layers

FP16 kept: Norm layers, embeddings, VAE

Offload: Enabled in both FP16 and W8A16

This ensures true quantization, not pseudo casting.

♻️ Reproducibility Rules

Same prompt & seed

Same resolution / frames / steps

Same offloading policy

Paired video comparison only

Metrics computed on real generated videos

📚 Citation

If you use this project, please cite:

Wan2.1-T2V-1.3B

OmniQuant

TorchAO

Video Diffusion Survey (Melnik et al.)

📬 Contact

Zhang Shenghe
City University of Macau
Email: D24091111148@cityu.edu.mo

⭐ Acknowledgement

This project is intended as a reproducible engineering baseline for diffusion PTQ research on consumer GPUs.

End of README
