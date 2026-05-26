# X-Former 模型实现详解

## 架构总览

```
输入 batch
  ├── image [B, 3, 64, 160, 160]    → ImageEncoder       → [B, 32, 256]
  ├── clinical [B, F]               → ClinicalEncoder    → [B, 8,  256]
  └── radiomics [B, T, 16]          → RadiomicsEncoder   → [B, 1,  256]
                                       (+ lesion_token)    [B, 1,  256]
         │
         ▼
    投影到统一维度 → concat → [B, 41, 256] → BottleneckFusion → [B, 8, 256]
         │
         ▼
    flatten [B,2048] ─┬─→ Classifier → logits [B,2]
                      └─→ GRL → DomainHead → domain_logits [B,2]
```

---

## 1. image_encoder.py — 图像编码器

### 方法

**Swin-UNETR (PretrainedSwin3D)**：第一篇论文的 3D Swin Transformer，使用窗口注意力提取脑影像特征。配置：embed_dim=48, window_size=(7,7,7), patch_size=(2,2,2), depths=(2,2,2,2)。`extract_features()` 返回最后层特征经过 Conv3d(768→512) 的输出。

**PerceiverCompressor**：K=32 个可学习潜在向量作为 query，对 Swin 输出的空间特征 `[B, 512, D, H, W]` 做交叉注意力压缩。展平空间维度→投影到 256 维→query=latent tokens, key/value=spatial tokens→输出固定数量 token `[B, 32, 256]`。

**消融备选 (use_perceiver=False)**：全局平均池化 `[B,512,D,H,W]→[B,512]`→Linear→[B,256]→unsqueeze 为 `[B,1,256]`。

### 冻结策略

```
freeze_layers=2:  patch_embed + layers1 + layers2  冻结 (底层通用特征)
                  layers3 + layers4 + proj_head     可训练 (高层语义)
freeze_layers=1:  patch_embed + layers1             冻结
                  layers2/3/4 + proj_head           可训练
```

### 维度流

```
input:          [B, 3, 64, 160, 160]
Swin output:    [B, 512, D, H, W]           (D/H/W 取决于 target_shape/16)
flatten space:  [B, S, 512]                 (S = D*H*W)
proj_in:        [B, S, 256]
cross-attn:     latents[B,32,256] query  key/value  spatial[B,S,256]
output:         [B, 32, 256]
```

---

## 2. clinical_encoder.py — 原型临床编码器

### 方法

**每个特征独立嵌入**：对每个临床特征位置（年龄、NIHSS 等），用独立的 `Linear(1, 128)` 将标量映射为向量。F 个特征 → F 个向量 `[B, F, 128]`。

**可学习原型注意力**：M=8 个可学习原型向量作为 query，F 个特征向量作为 key/value。原型通过缩放点积注意力聚合所有临床信息。

**缺失特征处理**：两个数据集临床列不统一，联合训练时补零对齐。特征 mask 将填充位置在 softmax 前设为 -inf，注意力自然忽略。

**数据集嵌入**：可选的 `Embedding(2, 128)`，为 MRI (id=0) 和 CT (id=1) 分别加一个可学习偏置，让模型感知数据来源。

### 维度流

```
input:          [B, F]                      F=15 (77sets) or 24 (ISLE)
per-feature:    [B, 1] × F → stack →       [B, F, 128]
prototype attn: Q=protos[B,8,128]  K/V=feats[B,F,128]
scores:         [B, 8, F]  → softmax
output:         attn @ feats = [B, 8, 128] + optional ds_embed
norm:           [B, 8, 128]
```

---

## 3. radiomics_encoder.py — 放射组学编码器

### 方法

**数据集特定投影**：77sets 16 维 → `Linear(16, 128)`，ISLE 15 维 → `Linear(15, 128)`。联合训练时逐样本路由（`is_77` / `is_isle` mask），避免跨数据集维度冲突。

**共享时序 Transformer**：2 层 TransformerEncoder (nhead=4)，学习灌注时间序列动态。前面加一个 CLS token，最后所有时间步取均值。

**病灶分支 (当前关闭)**：ISLE 独有的静态病灶形状特征（6 维），通过 2 层 MLP 编码为 token。无病灶时乘以零 mask。

### 维度流

```
input:          [B, T, 16]                  T=51 (77sets) or 41 (ISLE), 特征补零至 16
per-sample:     is_77 → Linear(16,128) / is_isle → slice[:,:15] → Linear(15,128)
output:         [B, T, 128]
+ CLS:          [B, T+1, 128]
Transformer:    [B, T+1, 128]
mean pool:      [B, 1, 128]

lesion (opt):   [B, N, 6] → MLP → [B, N, 128] → mean → [B, 1, 128]
```

---

## 4. bottleneck_fusion.py — 信息瓶颈融合（核心创新）

### 方法

**BottleneckLayer**（单层，重复 4 次）：
1. **自注意力**：所有 token 互相看（跨模态信息交换）
2. **压缩**：slots (B=8) 作为 query，所有 token 作为 key/value → 信息被压缩进槽位
3. **扩展**：所有 token 作为 query，slots 作为 key/value → 槽位信息广播回 token
4. **FFN**：token 过 2 层 MLP (dim → 4*dim → dim, GELU)

关键：压缩和扩展不对称——信息先被强制压入 8 个槽位，再从槽位回去。这个"窄口"就是信息瓶颈。

**BottleneckFusion**：初始化 B=8 个可学习槽位，叠 4 层 BottleneckLayer，最后 LayerNorm 输出。

### 维度流

```
input tokens:   [B, 41, 256]               (32 img + 8 cli + 1 rad = 41)
slots:          [B, 8, 256]                (可学习，随机初始化)

Layer 1-4, each:
  self-attn:    tokens [B,41,256] ← 自注意力
  compress:     slots [B,8,256] ← cross_attn(Q=slots, K/V=tokens)
  expand:       tokens [B,41,256] ← cross_attn(Q=tokens, K/V=slots)
  FFN:          tokens [B,41,256] ← Linear→GELU→Dropout→Linear→Dropout

output slots:   [B, 8, 256]
```

---

## 5. x_former.py — 主模型组装

### 结构

| 组件 | 实例 | 参数量 |
|------|------|--------|
| ImageEncoder | Swin(~20M) + Perceiver | ~20M (冻结 ~15M) |
| PrototypeClinicalEncoder | M=8 prototypes | ~0.1M |
| RadiomicsEncoder | 2 projectors + Transformer | ~0.3M |
| projection layers | 3× Linear(·, 256) | ~0.2M |
| BottleneckFusion | 4 layers × 3 attn + FFN | ~4M |
| Classifier | 2048→256→128→2 | ~0.5M |
| DomainHead | 2048→128→2 | ~0.3M |

**总计约 10M 可训练参数**

### 消融控制

| 开关 | 效果 |
|------|------|
| `use_bottleneck=false` | 均值池化替代瓶颈融合 |
| `use_perceiver=false` | 全局平均池化替代 Perceiver |
| `use_domain_loss=false` | 关闭域对抗 |
| `use_contrastive_loss=false` | 关闭对比损失 |
| `use_proto_clinical=false` | 2 层 MLP 替代原型编码器 |
| `use_lesion=false` | 不使用病灶放射组学 |
| `use_consistency_loss=false` | 关闭一致性损失 |

### forward 完整流程

```python
# 1. 编码
img  = ImageEncoder(image)                          # [B, 3,64,160,160] → [B, 32,256]
cli  = ClinicalEncoder(clinical, dataset_id, mask)  # [B, F] → [B, 8,128] → proj→ [B, 8,256]
rad  = RadiomicsEncoder(radiomics, dataset_id, ...) # [B, T,16] → [B, 1,128] → proj→ [B, 1,256]

# 2. 拼接
tokens = cat([img, cli, rad])                       # [B, 41, 256]

# 3. 瓶颈
bottleneck = BottleneckFusion(tokens)               # [B, 8, 256]
flat = bottleneck.flatten(1)                        # [B, 2048]

# 4. 输出
logits       = Classifier(flat)                     # [B, 2]
domain       = DomainHead(GRL(flat))                # [B, 2]  梯度反转
```
