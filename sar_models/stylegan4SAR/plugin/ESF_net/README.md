# ESF_net (Module-2 Prep)

该目录包含模块2前置代码（不改动原 StyleGAN 网络）：

1. `esf_model.py`：属性网络 `ESFAttributeNet`，输出 `KxD` 属性。
2. `esf_dataset.py`：从 `datasets/ascdata.zip` 直接读取训练数据。
3. `train_esf.py`：训练脚本，权重保存到 `plugin/ESF_net/ascweight/`。
4. `frozen_adapter.py`：冻结权重并产出注入前 embedding 的适配器。

## 训练命令

```bash
python -m plugin.ESF_net.train_esf
```

训练后权重：

- `plugin/ESF_net/ascweight/esf_attr_best.pt`
- `plugin/ESF_net/ascweight/esf_attr_last.pt`

## 冻结并用于注入前处理（示例）

```python
import torch
from plugin.ESF_net import build_frozen_esf_adapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adapter = build_frozen_esf_adapter(
    ckpt_path='plugin/ESF_net/ascweight/esf_attr_best.pt',
    device=device,
    embed_dim=128,
)

# x: [B,1,128,128]
# out['attrs']: [B,K,D]
# out['embedding']: [B,128]
out = adapter(x)
```

