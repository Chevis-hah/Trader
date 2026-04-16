# models/ - ML 模型存储

## 职责

存放训练好的 ML 模型文件 (pickle 格式)。

> v2.3 无本目录变更；仓库批次见 `SYNC_UPDATE_LOG.md`。

## 说明

- AlphaModel (alpha/ml_model.py) 训练后通过 `save()` 方法将模型序列化到此目录
- 模型文件格式: pickle
- 当前为空目录 (v3 未启用 ML 策略)
- 不纳入版本控制 (模型文件体积大且环境相关)
