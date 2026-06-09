# OmniVoice-Triton-API

[English](README.md)

基于 [OmniVoice](https://github.com/k2-fsa/OmniVoice) 的 OpenAI 兼容语音合成 API，使用 [omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton) 加速。

比原版 OmniVoice 快 3.4 倍。每句话约 170ms。646 种语言。声音克隆 + 语音设计。无需 flash-attn。

## 功能特性

- OpenAI 兼容的 `/v1/audio/speech` 接口（JSON body）
- **Triton 内核融合 + CUDA Graph** — 比原版快 3.4 倍
- 646 种语言（开源 TTS 中覆盖最广）
- 3 秒参考音频即可克隆声音
- 自然语言描述设计语音
- 内置语音预设（auto、female、male、child、elderly 等）
- 模型仅约 2GB，显存约 2GB
- 支持 RTX 50 系列（Blackwell）显卡

## 快速开始

```bash
docker run -d --gpus all \
  -p 8080:8080 \
  -v /mnt/user/appdata/omnivoice-triton-api/models:/root/.cache/huggingface \
  --shm-size=4g \
  --name omnivoice-triton-api \
  ghcr.io/hsiang-han/omnivoice-triton-api:latest
```

或使用 docker compose：

```bash
docker compose -f docker/gpu/docker-compose.yml up -d
```

首次启动下载模型（约 2GB）。国内用户设置 `HF_ENDPOINT=https://hf-mirror.com` 加速下载。

## 使用示例

```bash
# 自动语音
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "auto"}' \
  --output output.wav

# 语音设计（描述你想要的声音）
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好世界", "voice": "年轻女性，温暖柔和", "language": "zh"}' \
  --output designed.wav

# 使用预设
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "测试中文女声。", "voice": "female_zh"}' \
  --output female.wav

# 声音克隆
curl -X POST http://localhost:8080/v1/audio/speech/clone \
  -F "input=这是克隆的声音。" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=参考音频中说的话。" \
  --output cloned.wav

# 快速推理（16步代替32步）
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "快速模式！", "voice": "female_zh", "num_step": 16}' \
  --output fast.wav

# 查看预设音色
curl http://localhost:8080/v1/voices
```

## 语音预设

| 预设 | 描述 |
|------|------|
| auto | 自动生成语音（无指令） |
| female | 通用女声 |
| male | 通用男声 |
| female_en | 美式英语女声 |
| male_en | 美式英语男声 |
| female_zh | 中文女声 |
| male_zh | 中文男声 |
| child | 童声 |
| elderly | 老年女声 |

也可以传任意描述作为 `voice`（如 "温暖的中年男声，语速偏慢"）。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/audio/speech` | POST | 语音合成（JSON body，OpenAI 兼容） |
| `/v1/audio/speech/clone` | POST | 声音克隆（Form + 文件上传） |
| `/v1/voices` | GET | 列出可用语音预设 |
| `/v1/models` | GET | 列出模型 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 文档 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_ID | k2-fsa/OmniVoice | HuggingFace 模型 ID |
| RUNNER_MODE | hybrid | 推理模式：hybrid、triton、faster、base |
| NUM_STEPS | 32 | 扩散步数（16 更快，32 质量更好） |
| DTYPE | fp16 | 模型精度 |
| DEVICE | cuda:0 | CUDA 设备 |
| PORT | 8080 | API 端口 |
| HF_ENDPOINT | https://huggingface.co | HuggingFace 镜像地址 |

## 性能

omnivoice-triton 基准（RTX 5090）：

| 模式 | 加速 | 典型延迟 |
|------|------|---------|
| base | 1.0x | ~500ms |
| triton | ~1.5x | ~330ms |
| faster (CUDA Graph) | ~2.3x | ~220ms |
| **hybrid** (Triton + CUDA Graph) | **3.4x** | **~170ms** |

## 硬件要求

- NVIDIA 显卡，2GB+ 显存
- 驱动版本 550+（Ampere/Ada）或 570+（Blackwell）
- 安装 NVIDIA Container Toolkit 的 Docker 环境

## 致谢

- [OmniVoice](https://github.com/k2-fsa/OmniVoice) — k2-fsa (Next-gen Kaldi)，模型（646 种语言，Apache-2.0）
- [omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton) — [@newgrit1004](https://github.com/newgrit1004)，Triton 内核融合 + CUDA Graph 加速

## 许可证

Apache-2.0
