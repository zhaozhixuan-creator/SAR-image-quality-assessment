# ESF 模块接入变更说明

本次只完成“模块2的冻结权重注入判别器（D）”接线，未改动生成脚本逻辑。

## 修改文件

1. `training/networks.py`
- 在 `Discriminator` 中新增参数：
  - `esf_enabled`（是否启用 ESF 注入）
  - `esf_ckpt_path`（ESF 预训练权重路径）
  - `esf_fusion_weight`（ESF 分支融合权重）
- 启用时加载 `plugin/ESF_net/frozen_adapter.py` 的冻结模型。
- 在 `forward()` 末端增加 ESF 分支 logit，并与原判别器 logit 做加权相加。
- `esf_enabled=False` 时与原始判别器保持等价。

2. `train.py`
- 新增 CLI 参数：
  - `--esf_enabled`
- 其余 ESF 参数（权重路径、融合权重）统一从 `plugin/ESF_net/ESF_config.py` 读取，不再提供 CLI 覆盖。
- 将配置参数写入 `args.D_kwargs` 传递给判别器构造函数。

## 使用方式

示例（在原训练命令基础上增加 ESF 参数）：

```bash
python train.py \
  --outdir=training-runs \
  --data=datasets/mstar15_5class_stride5_maxdev-128.zip \
  --gpus=1 --cond=1 --cfg=auto --aug=ada --batch=32 --kimg=5000 --snap=25 \
  --esf_enabled=1
```

对应配置位置：
- `plugin/ESF_net/ESF_config.py`
  - `d_inject_ckpt`
  - `d_fusion_weight`
