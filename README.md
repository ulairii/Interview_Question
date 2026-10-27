# ML Coding 面试准备

面向 ML / CV / 自动驾驶方向的 ML Coding 面试：把数学公式写成**向量化、数值稳定、无 bug** 的 NumPy / PyTorch 代码，并能讲清复杂度与边界条件。

## 仓库结构

```
.
├── README.md                 ← 你在这里
├── notes/                    ← 讲解笔记（每个主题一个文件，含标准答案和「我的笔记」留白区）
│   ├── guide/                  准备指南：方法、易错速查、复习顺序、代码订正
│   ├── classic_ml/             经典 ML：KNN / K-Means / 回归 / 决策树 / GMM / PCA
│   ├── dl_components/          DL 组件：Attention（7 篇）/ 归一化 / 卷积 / Dropout & LoRA
│   ├── losses_metrics/         损失与指标：Chamfer / 交叉熵 & Focal & InfoNCE / AUC / IoU & NMS
│   └── sampling/               采样与解码
├── code/
│   ├── ml_coding.py            全部参考实现（带类型注解和 shape 注释）
│   └── test_ml_coding.py       对照 PyTorch / torchvision / 暴力解的测试
├── tools/
│   └── sync_answers.py         由 code/ml_coding.py 重新生成各笔记里的「标准答案」折叠块
└── practice/                 ← 模拟面试
    ├── exercises.py            空白题（在这里作答）
    ├── template/               干净模板（reset 用）
    └── run.py                  list / mock / check / reset
```

## 怎么用

1. **学**：按下面的学习路线读 `notes/`，每篇顶部有进度勾选框，末尾有「✏️ 我的笔记」。
2. **练**：在 `practice/exercises.py` 里手写，`python run.py check <题名>` 判分。
3. **模拟**：`python run.py mock` 随机抽题，限时，交卷后自动判分并给出追问清单。

```bash
cd practice
python run.py list            # 18 道题 + 建议时长
python run.py mock            # 随机抽题，计时
python run.py mock kmeans     # 指定题目
python run.py check all       # 检查 exercises.py 里已写的所有题
python run.py reset           # 恢复成空白
```

验证参考实现：`cd code && CUDA_VISIBLE_DEVICES= python test_ml_coding.py`（纯 CPU）。

## 学习路线

| 天 | 内容 |
|---|---|
| 1 | Chamfer、KNN、K-Means、IoU / NMS |
| 2 | 逻辑回归、交叉熵、Focal / InfoNCE、AUC |
| 3 | Attention（总览 → SDPA → Self → Cross → MultiHead → KV Cache）、归一化、Dropout、LoRA |
| 4 | 决策树 / 随机森林、GMM、PCA、Conv2d |
| 5 | 采样与解码；全部默写一遍，每题限时，口述追问 |

自测标准：**能默写 → 能说复杂度 → 能说 3 个边界条件 → 能说一个优化**。详见 [复习顺序](notes/guide/02_复习顺序.md)。

## 目录与总进度

### 准备指南
- [ ] [考什么、怎么准备（含代码规范与写法风格）](notes/guide/00_方法与流程.md)
- [ ] [速查：最容易写错的地方](notes/guide/01_易错速查.md)
- [ ] [建议复习顺序](notes/guide/02_复习顺序.md)
- [ ] [对原材料代码的订正](notes/guide/03_代码订正.md)

### 经典机器学习
| 主题 | 笔记 | 练习题名 |
|---|---|---|
| KNN | [01_KNN](notes/classic_ml/01_KNN.md) | `knn` |
| K-Means → K-Means++ | [02_KMeans](notes/classic_ml/02_KMeans.md) | `kmeans` |
| Linear / Logistic Regression | [03_线性与逻辑回归](notes/classic_ml/03_线性与逻辑回归.md) | `logreg` |
| 决策树 / 随机森林 | [04_决策树与随机森林](notes/classic_ml/04_决策树与随机森林.md) | `tree` |
| GMM（EM） | [05_GMM](notes/classic_ml/05_GMM.md) | — |
| PCA | [06_PCA](notes/classic_ml/06_PCA.md) | `pca` |

### 深度学习组件
| 主题 | 笔记 | 练习题名 |
|---|---|---|
| Attention 总览 | [00_总览](notes/dl_components/attention/00_总览.md) | — |
| Scaled Dot-Product 与 Mask | [01_SDPA与Mask](notes/dl_components/attention/01_SDPA与Mask.md) | `attention` |
| Self-Attention | [02_SelfAttention](notes/dl_components/attention/02_SelfAttention.md) | `attention` |
| Cross-Attention | [03_CrossAttention](notes/dl_components/attention/03_CrossAttention.md) | `attention` |
| Multi-Head | [04_MultiHead](notes/dl_components/attention/04_MultiHead.md) | `mha` |
| KV Cache | [05_KVCache](notes/dl_components/attention/05_KVCache.md) | — |
| Attention 追问与验证 | [06_追问与验证](notes/dl_components/attention/06_追问与验证.md) | — |
| BatchNorm / LayerNorm / RMSNorm | [01_BN_LN_RMSNorm](notes/dl_components/01_BN_LN_RMSNorm.md) | `norm` |
| Conv2d / MaxPool | [02_Conv2d_MaxPool](notes/dl_components/02_Conv2d_MaxPool.md) | `conv` |
| Dropout / LoRA | [03_Dropout_LoRA](notes/dl_components/03_Dropout_LoRA.md) | `dropout`, `lora` |

### 损失函数与评估指标
| 主题 | 笔记 | 练习题名 |
|---|---|---|
| Chamfer Distance | [01_Chamfer_Distance](notes/losses_metrics/01_Chamfer_Distance.md) | `chamfer` |
| Cross-Entropy / Focal / InfoNCE | [02_损失函数](notes/losses_metrics/02_损失函数.md) | `ce`, `focal_infonce` |
| AUC-ROC | [03_AUC](notes/losses_metrics/03_AUC.md) | `auc` |
| IoU / GIoU / NMS | [04_IoU_GIoU_NMS](notes/losses_metrics/04_IoU_GIoU_NMS.md) | `iou`, `nms` |

### 采样与解码
| 主题 | 笔记 | 练习题名 |
|---|---|---|
| Temperature / Top-k / Top-p | [01_采样与解码](notes/sampling/01_采样与解码.md) | `sampling` |

## 笔记的统一结构

每个主题文件：**题目与思路 → 分步代码（一行一个想法、行尾标 shape）→ 追问表 → ✅ 标准答案（默认折叠）→ ✏️ 我的笔记**。
代码风格：全部带类型注解；docstring 写明输入输出的 shape 与 dtype；边界条件单独写 `if` 并注释。
