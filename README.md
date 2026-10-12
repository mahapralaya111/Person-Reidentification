# 抗恶劣环境行人重识别系统

基于深度学习的行人重识别（Person Re-ID）系统，面向灾害应急搜救场景：在烟雾、雨雾、低光照等恶劣环境下，跨摄像头识别和追踪特定人员。

起因是在了解应急救援时注意到一个问题：搜救现场常用的人脸识别基本失效——被困人员往往背对/侧对镜头、面部有遮挡，或者姿态是倒地、匍匐。而行人重识别靠的是整体外观特征（体型、衣着、步态），恰好绕开了这些限制。所以做了这个系统，把 ReID 从"安防找嫌疑人"迁移到"搜救找被困人员"的场景，并针对恶劣环境做了专门的数据增强和训练策略。

## 核心功能

- **人员检索**：上传目标照片，在底库（gallery）中检索最相似的人员，输出 Rank-K 排序结果
- **恶劣环境增强**：模拟雨、雾、低光照、运动模糊、姿态畸变等条件的训练时增强，提升模型在真实灾害场景下的鲁棒性
- **底库增强**：对少量底库照片（每人 15~30 张）做多视角/多环境扩充，缓解小样本过拟合
- **视频流处理**：YOLOv8 行人检测 + 多目标跟踪 + ReID 检索，支持对已有视频文件做时间线识别报告
- **双模态支持**：模型结构上预留可见光/红外特征融合（红外部分仍在开发中）
- **多种使用方式**：CLI / Streamlit Web 界面 / Flask REST API / Tkinter GUI 启动器

## 模型与效果

骨干网络支持 OSNet、ResNet50/34、ResNet-IBN、AGW、PCB，损失函数支持 CrossEntropy + Triplet / Circle Loss。

Market-1501 上的评估结果（单卡 1080Ti 级别 GPU）：

| 配置 | Rank-1 | mAP |
|------|--------|-----|
| OSNet | 85.2% | 72.1% |
| ResNet50 | 87.5% | 74.3% |
| OSNet + Circle Loss | 88.1% | 76.5% |
| OSNet + Circle Loss + 环境增强 | **89.3%** | **78.2%** |

恶劣环境下的对比测试（低光照条件下标准模型掉点非常严重）：

| 环境条件 | 标准模型 | 增强模型 | 提升 |
|----------|----------|----------|------|
| 正常光照 | 88.1% | 89.3% | +1.2% |
| 低光照 | 45.3% | 76.8% | **+31.5%** |

## 快速开始

### 环境要求

- Python 3.8 ~ 3.10
- PyTorch >= 1.8（GPU 版需要 CUDA >= 11.0，CPU 也能跑推理）
- 可选：NVIDIA GPU（训练建议）、ultralytics（视频处理的人体检测）

```bash
git clone https://github.com/mahapralaya111/Person-Reidentification.git
cd Person-Reidentification
pip install -r requirements.txt

# 环境自检
python main.py check
```

Windows 用户可以直接双击 `安装即用.bat`，脚本会自动创建虚拟环境并安装依赖。

### 训练

```bash
# 完整训练（OSNet + Circle Loss + 恶劣环境增强 + 余弦退火）
python main.py train --backbone osnet --use_circle_loss --use_weather_aug --use_cosine_annealing --epochs 100

# 没有数据集时可以先跑 demo 数据体验流程
python main.py train --demo --epochs 5

# Market-1501 专用训练脚本（支持自动下载）
python src/train_market1501.py
```

数据集组织为 Market-1501 格式（`bounding_box_train/`、`bounding_box_test/`、`query/`），自定义数据可以用 `tools/prepare_train_data.py` 一键整理。

### 人员检索

```bash
# 构建底库特征
python main.py inference --mode gallery --images rescue_gallery --output gallery.npz

# 用查询图在底库中搜索
python main.py inference --mode search --query query.jpg --images rescue_gallery --top_k 10
```

底库照片按 `底库目录/人员ID/xxx.jpg` 组织，每人 15~30 张效果比较稳。

### 视频识别

```bash
python main.py video --video test.mp4 --gallery gallery.npz
```

会输出逐帧的检测/跟踪/识别结果，并生成时间线 CSV 报告（见 `reports/`）。

### Web 界面 / API

```bash
python main.py web    # Streamlit，默认 http://localhost:8501
python main.py api    # Flask REST API，默认 http://localhost:5000
```

API 文档见 [docs/API文档.md](docs/API文档.md)，接口支持图片检索、底库管理、健康检查等。

### GUI 启动器

```bash
python launcher.py
```

Tkinter 写的图形启动器，集成环境检查、一键训练、底库整理、视频识别等功能，适合不想敲命令行的场景。

## 目录结构

```
├── main.py                 # 统一 CLI 入口（train/inference/web/api/video/eval/check）
├── launcher.py             # Tkinter GUI 启动器
├── src/
│   ├── backbone.py         # 骨干网络：OSNet/ResNet/AGW/PCB/IBN
│   ├── train.py            # 通用训练器（多数据集/多损失/学习率策略）
│   ├── train_market1501.py # Market-1501 训练脚本
│   ├── eval.py             # CMC / mAP / mINP / 重排序
│   ├── inference.py        # 特征提取与底库检索
│   ├── augmentation.py     # 恶劣环境数据增强
│   ├── augment_gallery.py  # 底库增强
│   ├── video_processor.py  # 视频流处理与多目标跟踪
│   └── config/             # YAML 配置管理
├── web/                    # Streamlit 界面 + Flask API
├── configs/                # 训练/推理 YAML 配置
├── tools/                  # 数据集整理、评估、检查脚本
├── docs/                   # API、训练、部署文档
├── eval_results/           # 评估结果与图表
└── reports/                # 视频识别时间线报告示例
```

## 开发记录

整个项目大致分四步走的：

1. **跑通基线**：先在 Market-1501 上把 OSNet 训练和 CMC/mAP 评估流程跑通（tag `market1501-baseline`）
2. **恶劣环境增强**：实现雨雾/低光/运动模糊等增强，这一步对低光照场景的提升最明显（tag `weather-augmentation`）
3. **走向视频**：接入 YOLOv8 检测和跟踪，把静态图检索扩展到视频流（tag `video-reid`）
4. **可用性**：Web 界面、REST API、GUI 启动器，让它变成一个能演示、能被其他系统调用的完整服务（tag `web-api-release`）

## 踩坑记录

一些印象比较深的问题，供参考：

- **Market-1501 的 gt_bbox 标注有重复/冲突 ID**，直接拿它重建训练集会导致训练集和查询集身份重叠，评估结果虚高。后来写了 `tools/rebuild_market1501.py` 按规则重建并做冲突检查。
- **低光照场景标准模型几乎不可用**（Rank-1 从 88% 掉到 45%）。单纯调 gamma 不够，最终是靠训练时随机叠加低光 + 噪声 + 运动模糊的组合增强解决的。
- **底库只有每人十几张照片时直接微调容易过拟合**，采用"Market-1501 预训练 + 小学习率底库微调 + 底库增强"的三段式策略，比直接训练稳定很多。
- **视频场景 YOLO 检出的框和 ReID 训练分布差异大**（检测框更松、背景更多），测试时需要对底库和查询图用一致的检测/裁剪流程，否则检索精度会明显下降。

## 局限与 TODO

- [ ] 红外模态融合还未完成，目前主要是结构预留
- [ ] 跨数据集泛化没有系统评估（Market-1501 -> 实际搜救场景存在 domain gap）
- [ ] 实时视频流（RTSP）支持，目前以视频文件为主
- [ ] 模型权重没有随仓库发布（体积原因），需要自行训练或下载 Market-1501 预训练权重

## 说明

- 仓库不含数据集、模型权重与私人底库照片，数据集请自行下载 Market-1501
- 本项目仅用于学术研究与应急搜救技术验证，请勿用于侵犯个人隐私的用途

## License

[MIT](LICENSE)
