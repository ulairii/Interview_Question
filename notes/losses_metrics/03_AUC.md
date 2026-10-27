# AUC-ROC

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


物理意义：随机取一个正样本和一个负样本，正样本得分更高的概率（并列算 0.5）。

**排序 + 累积 + 梯形（处理并列）**：

```python
n_pos = int((y == 1).sum()); n_neg = len(y) - n_pos
# 1. 按分数从高到低排序
order = np.argsort(-score, kind='stable')
y_sorted = y[order]                                # (N,)
s_sorted = score[order]                            # (N,)
# 2. 阈值位置：每个不同分数的最后一个下标（并列的分数一起移动）
change = np.where(np.diff(s_sorted))[0]            # 分数发生变化的位置
cut = np.r_[change, len(y_sorted) - 1]             # (T,)
# 3. 每个阈值下的计数
tps = np.cumsum(y_sorted)[cut]                     # (T,) 被判为正的真正例
fps = (cut + 1) - tps                              # (T,) 被判为正的负例
# 4. 比率，补上起点 (0, 0)
tpr = np.r_[0, tps] / n_pos                        # (T+1,)
fpr = np.r_[0, fps] / n_neg                        # (T+1,)
# 5. 梯形面积
widths = np.diff(fpr)                              # (T,)
heights = (tpr[:-1] + tpr[1:]) / 2                 # (T,)
auc = np.sum(widths * heights)
```

**秩和法**（无并列时）：

```python
ranks = np.argsort(np.argsort(score)) + 1
auc = (ranks[y==1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
```

- 有并列时秩和法要用平均秩（`scipy.stats.rankdata`），否则偏差；累积法天然正确。
- 边界：全正或全负 → AUC 无定义，返回 0.5 / 抛错并说明。
- 复杂度 O(N log N)，瓶颈是排序。
- PR 曲线同理，但在类别极不平衡时比 ROC 更敏感。
- 代码已与 O(N²) 暴力解（含并列）及 sklearn 对照。


<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Sort + cumsum + trapezoid. Handles ties.

    Args:
        y_true: (N,) in {0, 1}.
        y_score: (N,) float, higher = more positive.
    Returns:
        AUC in [0, 1]; 0.5 if only one class is present.
    """
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # 1. sort by score, high to low
    order = np.argsort(-y_score, kind='stable')
    y_sorted = y_true[order]                                    # (N,)
    s_sorted = y_score[order]                                   # (N,)
    # 2. thresholds: last index of each distinct score (so ties move together)
    change = np.where(np.diff(s_sorted))[0]                     # positions where the score changes
    cut = np.r_[change, len(y_sorted) - 1]                      # (T,)
    # 3. counts at each threshold
    tps = np.cumsum(y_sorted)[cut]                              # (T,) positives predicted positive
    fps = (cut + 1) - tps                                       # (T,) negatives predicted positive
    # 4. rates, with the (0, 0) starting point
    tpr = np.r_[0, tps] / n_pos                                 # (T+1,)
    fpr = np.r_[0, fps] / n_neg                                 # (T+1,)
    # 5. trapezoid area under the curve
    widths = np.diff(fpr)                                       # (T,)
    heights = (tpr[:-1] + tpr[1:]) / 2                          # (T,)
    return float(np.sum(widths * heights))


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U. Correct only when there are no tied scores.

    Args/Returns: same as auc_roc (assumes both classes present).
    """
    ranks = np.argsort(np.argsort(y_score)) + 1
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



