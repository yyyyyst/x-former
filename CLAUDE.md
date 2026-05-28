# X-Former 项目规则

## 实验管理

- 每次改配置后必须更新 `config/experiments.md`：记录版本号、日期、日志文件名、改动参数、per-fold 结果、汇总对比、分析
- 版本号递增（V1-V7...），日志文件自动带时间戳
- 训练命令：`bash run.sh --gpus "2" --nohup`
- 查看结果：`grep -E "Fold.*best|test AUC|FINAL|pooled" logs/xformer_*.log`

## 消融开关

config/config.yaml 中 `ablation` 字段控制组件开关，代码自动读取：

```
use_bottleneck      → 信息瓶颈融合 / 均值池化 (默认 true)
use_perceiver       → Perceiver 压缩 / 全局平均池化
use_domain_loss     → 域对抗损失 (当前 false, V7)
use_contrastive_loss → 对比损失 (当前 true, weight=0.01)
use_consistency_loss → 一致性损失 (依赖病灶，默认 false)
use_lesion           → 病灶放射组学 (默认 false)
use_proto_clinical   → 原型编码器 / MLP
joint_training       → 联合训练 / 独立训练
```

## 调参经验总结

1. **contrastive weight 不能 > 0.03** — 小 batch (≤8) 下对比损失数值比 focal 大数倍，会主导梯度
2. **model 不能太大** — 77sets 每折 ~50 样本，解冻 layer2 (+8M 参数) 直接过拟合
3. **bottleneck 不能 < 8 slots** — 到 4 个就欠拟合
4. **域对抗和瓶颈冲突** — GRL 让模型在两个数据集间二选一，瓶颈 B=8 本身已跨模态对齐
5. **77sets MRI 不需要 z-score** — CT 窗宽在第一篇里直接用了且跑出 0.813，改 z-score 反而降了

## 关键发现（截至 V6）

- 最佳版本：V5（无对比、有域对抗），77sets 0.805 pooled，距第一篇 -0.008
- 域对抗损失导致 fold 间 77sets/ISLE 反相关（一涨一跌）
- V7 测试：关域对抗，纯靠瓶颈 + 极轻对比

## 文件结构

```
model/         — 模型组件
data_provider/ — 数据加载
train/         — 训练循环 + CV
losses/        — 损失函数
utils/         — 指标/绘图/Bootstrap
config/        — 配置 + 实验记录 + 文档
datasets/      — 数据（不提交 git）
pretrained/    — 预训练权重（不提交 git）
results/       — 训练输出（不提交 git）
logs/          — 训练日志（不提交 git）
```
