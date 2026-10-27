# Chamfer Distance（3D 视觉必考）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


### 1.1 定义

$$d_{CD}(P,Q)=\frac1{|P|}\sum_{p\in P}\min_{q\in Q}\|p-q\|_2^2+\frac1{|Q|}\sum_{q\in Q}\min_{p\in P}\|q-p\|_2^2$$

### 1.2 实现（面试版）

```python
def pairwise_sq_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    p_sq = np.sum(P ** 2, axis=1)                  # (N,)   ||p||²
    q_sq = np.sum(Q ** 2, axis=1)                  # (M,)   ||q||²
    cross = P @ Q.T                                # (N, M) p·q
    d = p_sq[:, None] + q_sq[None, :] - 2 * cross  # (N, M)
    return np.maximum(d, 0.0)                      # 夹紧浮点负数


def chamfer_distance(P: np.ndarray, Q: np.ndarray, squared: bool = True) -> float:
    d = pairwise_sq_dist(P, Q)                     # (N, M)
    if not squared:
        d = np.sqrt(d)
    p_to_q = d.min(axis=1)                         # (N,) 每个 p 到 Q 的最近距离
    q_to_p = d.min(axis=0)                         # (M,) 每个 q 到 P 的最近距离
    return float(p_to_q.mean() + q_to_p.mean())
```

- 最直观写法（先写这个也行）：`d = np.sum((P[:,None,:] - Q[None,:,:])**2, axis=-1)`，中间张量 `(N,M,D)`。
- Batch 版：`P:(B,N,D)`、`Q:(B,M,D)`，`P @ Q.swapaxes(1,2)` → `(B,N,M)`，`min(axis=2).mean(axis=1) + min(axis=1).mean(axis=1)`，见 `batched_chamfer_distance`。
- 一行 einsum：`np.einsum('bnd,bmd->bnm', P, Q)`。

### 1.3 Follow-up：为什么这样设计

**① 为什么用 min：** 点云是无序集合（置换不变），`P`、`Q` 点数还可以不同，没有"第 i 点对第 i 点"的对应关系。`min` 用最近邻作为**隐式对应**（贪心近似）。

**② 为什么双向：** 两个方向分别对应 Precision / Recall。

| 方向 | 含义 | 只留这一项会怎样 |
|---|---|---|
| P→Q | 精确性：预测点都落在真实表面上 | 所有预测点堆在 Q 的同一个点上，loss=0（坍缩） |
| Q→P | 完整性：真实表面各处都被预测点覆盖 | 预测点铺满整个空间，loss≈0（噪点泛滥） |

两项都要，才能既贴合表面又覆盖完整，同时满足对称性 `d(P,Q)=d(Q,P)`。代码里 `test_ml_coding.py` 有坍缩反例的验证。

**③ 为什么 mean 再相加：** 两边各自是"单点平均误差"，量纲对齐；用 sum 则点数多的一方权重自动变大（8192 vs 1024 差 8 倍），偏向 Recall；mean 还让 loss 与梯度量级不随采样点数变化，换点数不用重调学习率。**面试时主动确认：用 sum 还是 mean、是否平方。** 论文约定不一。

**④ 局限与扩展：** `min` 允许多对一，对密度不均不敏感 → EMD（强制一一匹配/双射，精确解 O(N³)，常用 Sinkhorn 近似）。

### 1.4 D 是什么 / 广播怎么对齐

- `D` 是点的坐标维，3D 点云 `D=3`；`P` 是形状 `(N,3)` 的二维数组，每行一个 `[x,y,z]`（不是 tuple 数组）。`P.shape` 才是 tuple。
- 代码不依赖 D：2D 点、6D（带法向量）、512 维特征都一样。

**广播规则**：shape 从右向左对齐，每一维要么相等，要么其中一个是 1（会被复制扩展）。

```
P[:, None, :]  (N, 1, 3)
Q[None, :, :]  (1, M, 3)
→ diff         (N, M, 3)    diff[i,j] = P[i] - Q[j]
```

口诀：**保护特征维（最后一维 3 必须对齐）、错开样本维（N 与 M 用 1 错开）、消掉特征维（`sum(axis=-1)`）**。

| 写法 | 结果 |
|---|---|
| `P[:,None,:] - Q[:,None,:]` | N≠M 直接报错；N=M 时静默只算 `P[i]-Q[i]`，漏掉交叉配对 |
| `P[:,:,None] - Q[None,:,:]` | `(N,3,1)` vs `(1,M,3)`，倒数第二维 3 vs M；M≠3 报错，**M=3 时不报错但完全算错** |
| `sum(axis=1)`（而非 -1） | 把 Q 的点维加掉，得到 `(N,3)`，不是距离矩阵 |

最危险的是"静默算错"（维度碰巧相等），所以每行都注释 shape。

### 1.5 matmul 展开：内存与数值

- 广播 `(N,M,D)` 中间张量：N=M=1e4, D=3 已是 3e8 个浮点；matmul 展开只需 `(N,M)`，且走 BLAS。
- `||p||²+||q||²-2p·q` 在 p≈q 时会出现 `-1e-16`，必须 `np.maximum(…, 0)`，否则 `sqrt` 得 NaN。
- 展开法在大坐标偏移时有精度损失（相消），可先对两个点云减去同一个均值。
- N、M 极大（>1e5）时，分块计算或用 KD-tree（`scipy.spatial.cKDTree`）求最近邻，避免 `(N,M)` 矩阵。

<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np


def pairwise_sq_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """All pairwise squared Euclidean distances via ||p-q||^2 = ||p||^2 + ||q||^2 - 2 p.q

    Args:
        P: (N, D) float.
        Q: (M, D) float.
    Returns:
        (N, M) float, entry [i, j] = ||P[i] - Q[j]||^2.
    """
    p_sq = np.sum(P ** 2, axis=1)                               # (N,)   ||p||^2
    q_sq = np.sum(Q ** 2, axis=1)                               # (M,)   ||q||^2
    cross = P @ Q.T                                             # (N, M) p . q
    d = p_sq[:, None] + q_sq[None, :] - 2 * cross               # (N, M)
    d = np.maximum(d, 0.0)                                      # fp error can give -1e-16
    return d


def chamfer_distance(P: np.ndarray, Q: np.ndarray, squared: bool = True) -> float:
    """Chamfer distance = mean(P->Q) + mean(Q->P).

    Args:
        P: (N, D) float, first point set.
        Q: (M, D) float, second point set.
        squared: use squared Euclidean distance if True.
    Returns:
        Scalar distance (float).
    """
    d = pairwise_sq_dist(P, Q)                                  # (N, M)
    if not squared:
        d = np.sqrt(d)
    p_to_q = d.min(axis=1)                                      # (N,) nearest Q point for each p
    q_to_p = d.min(axis=0)                                      # (M,) nearest P point for each q
    return float(p_to_q.mean() + q_to_p.mean())
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



