# K-Means → K-Means++

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍

完整代码：[code/ml_coding.py](../../code/ml_coding.py) 的 `KMeans`。

**面试节奏：先写普通 K-Means（阶段 1，约 8 分钟）→ 写通 → 再说"初始化可以改进"并加 K-Means++（阶段 2，约 4 分钟）。**
不要一上来就写 K-Means++，面试官会觉得你在绕。

---

## 算法一句话

重复两步直到中心不再移动：**① 每个点归到最近的中心；② 每个中心移到它那一簇的均值。**
目标：最小化簇内平方和 `SSE = Σ ||x_i − c_{label(i)}||²`。

---

# 阶段 1：普通 K-Means（随机初始化）

## 1.0 先写骨架（30 秒）

```python
class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4): ...
    def _assign(self, X): ...            # 每个点 → 最近中心
    def _update(self, X, labels): ...    # 每个中心 → 簇内均值
    def fit(self, X): ...                # 初始化 + 循环 + 收敛
    def predict(self, X): ...            # 分配一遍
```

## 1.1 距离矩阵（先写好这个工具函数）

目标：`d[i, j] = ||X[i] − C[j]||²`，shape `(N, K)`。和 Chamfer、KNN 是同一个函数：

```python
def pairwise_sq_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    p_sq = np.sum(P ** 2, axis=1)                  # (N,)
    q_sq = np.sum(Q ** 2, axis=1)                  # (K,)
    cross = P @ Q.T                                # (N, K)
    d = p_sq[:, None] + q_sq[None, :] - 2 * cross  # (N, K)
    return np.maximum(d, 0.0)                      # 夹紧浮点负数
```

一时想不起来就先写广播版：`np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)`。

## 1.2 完整代码（面试写这个）

```python
class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-4) -> None:
        self.k, self.max_iters, self.tol = k, max_iters, tol

    def _assign(self, X: np.ndarray) -> np.ndarray:
        d = pairwise_sq_dist(X, self.centroids)            # (N, K)
        return d.argmin(axis=1)                            # (N,)

    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new = np.empty_like(self.centroids)                # (K, D)
        for j in range(self.k):
            members = X[labels == j]                       # (n_j, D)
            if len(members) == 0:                          # 空簇：mean([]) 会得 NaN
                new[j] = X[np.random.randint(len(X))]      # 随机重放一个点
            else:
                new[j] = members.mean(axis=0)              # (D,)
        return new

    def fit(self, X: np.ndarray) -> "KMeans":
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()                     # (K, D) 随机选 K 个样本做初始中心
        for _ in range(self.max_iters):
            labels = self._assign(X)                       # (N,)
            new = self._update(X, labels)                  # (K, D)
            shift = np.linalg.norm(new - self.centroids)   # 中心移动量
            self.centroids = new
            if shift < self.tol:                           # 几乎不动 → 收敛
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(X)
```

只有 4 个部分：**初始化（2 行）→ 分配（2 行）→ 更新（含空簇）→ 收敛判断**。

## 1.3 一次迭代的手算例子

1 维，`X=[1, 2, 9, 10]`，`C=[0, 5]`：
- 分配：1→0, 2→0, 9→1, 10→1，`labels=[0,0,1,1]`
- 更新：`C=[1.5, 9.5]`
- 再分配不变 → shift=0 → 收敛。

## 1.4 阶段 1 的边界条件（说出来加分）

| 情况 | 处理 |
|---|---|
| 空簇 | 随机重放一个点（或取离所有中心最远的点），不处理会 NaN |
| `K > N` | `choice(replace=False)` 会报错，需限制或提前检查 |
| 距离出现 -1e-16 | `np.maximum(d, 0)` |
| 不收敛 | `max_iters` 兜底 |

## 1.5 面试官会接着问："随机初始化有什么问题？"

- 初始中心可能挤在一起 → 收敛慢，或陷入很差的**局部最优**；
- 每次结果不一样。

→ 引出阶段 2。

---

# 阶段 2：升级为 K-Means++

## 2.1 思路

让初始中心**彼此尽量远**：
1. 第一个中心：均匀随机选一个样本；
2. 之后每个中心：每个点被选中的概率 ∝ **它到已选最近中心的平方距离**（越远越可能被选）。

## 2.2 只改一处：初始化

阶段 1 里 `fit` 的这两行

```python
idx = np.random.choice(len(X), self.k, replace=False)
self.centroids = X[idx].copy()
```

换成 `self.centroids = self._init_plusplus(X)`，其余（`_assign` / `_update` / 循环）**一行不动**。新增：

```python
def _init_plusplus(self, X: np.ndarray) -> np.ndarray:
    N, D = X.shape
    centroids = np.empty((self.k, D), dtype=X.dtype)       # (K, D)
    centroids[0] = X[np.random.randint(N)]                 # 第一个：均匀随机
    for i in range(1, self.k):
        d = pairwise_sq_dist(X, centroids[:i])             # (N, i) 到已选中心的距离
        nearest = d.min(axis=1)                            # (N,)  到最近已选中心
        total = nearest.sum()
        if total > 0:
            probs = nearest / total                        # (N,)  归一化成概率
        else:                                              # 所有点都与已选中心重合
            probs = np.full(N, 1.0 / N)
        centroids[i] = X[np.random.choice(N, p=probs)]
    return centroids
```

逐行理解：

| 行 | 含义 |
|---|---|
| `d = pairwise_sq_dist(X, centroids[:i])` | 每个点到每个"已选"中心的距离，只用前 i 个 |
| `nearest = d.min(axis=1)` | 每个点离**最近的**已选中心有多远 |
| `probs = nearest / total` | 离得越远概率越大；已被选中的点距离是 0，概率 0，不会重复选 |
| `if total > 0 ... else` | 所有点重合时分母为 0，退化为均匀（只加 `1e-12` 会得到全 0 的概率向量，`choice` 报错） |

## 2.3 追问

| 追问 | 回答 |
|---|---|
| 为什么按距离平方而不是距离 | 更偏向远点，实践中效果更好，且理论上有 O(log K) 近似比保证 |
| 会不会选到离群点 | 有可能，概率更高；这是已知缺点 |
| 为什么还需要多次运行 | K-Means++ 仍是随机的，常取多次 SSE 最小的（`n_init`） |
| 初始化的额外代价 | O(N·K·D)，与一轮 K-Means 同量级 |

---

# 通用追问（两阶段都适用）

| 追问 | 回答 |
|---|---|
| 为什么一定收敛 | 分配与更新都不增大 SSE，且分配方案有限；但只到**局部最优** |
| 怎么缓解局部最优 | K-Means++；多次随机初始化取 SSE 最小 |
| 复杂度 | 每轮 O(N·K·D)；广播版显存 O(N·K·D)，matmul 展开 O(N·K) |
| N=1e5, K=100, D=512 | 广播版 5e9 浮点（约 40 GB），必须 matmul 展开 |
| 怎么选 K | 肘部法（SSE vs K）、轮廓系数、Gap statistic |
| 为什么要标准化 | 距离对尺度敏感，大尺度特征会主导 |
| 不适合什么数据 | 非球形簇、大小/密度差异大、有离群点（→ GMM / DBSCAN / K-Medoids）|
| 和 GMM 的关系 | K-Means 是 GMM 在 `Σ=σ²I, σ→0` 下的硬分配极限 |
| 数据量很大 | Mini-batch K-Means：每次用一小批更新中心 |
| 更新步想全向量化 | `one_hot = np.eye(K)[labels]`；`sums = one_hot.T @ X`；`counts = one_hot.sum(0)`；`sums / np.maximum(counts[:, None], 1)`（空簇另处理）|

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


class KMeans:
    """K-Means. fit(): init -> [assign -> update -> check] x iters.

    init="random" picks K random samples; init="kmeans++" spreads the initial centers out.
    Randomness comes from np.random (call np.random.seed(...) first if you need reproducibility).
    """

    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4,
                 init: str = "kmeans++") -> None:
        self.k, self.max_iters, self.tol, self.init = k, max_iters, tol, init
        self.centroids: np.ndarray                              # (K, D), set by fit

    # ---- step 1a: random initialisation ----
    def _init_random(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (K, D), K distinct random samples."""
        idx = np.random.choice(len(X), self.k, replace=False)   # (K,)
        return X[idx].copy()

    # ---- step 1b: K-Means++ initialisation ----
    def _init_plusplus(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (K, D). Next center is drawn with prob ∝ squared distance to nearest chosen center."""
        N = len(X)
        centroids = [X[np.random.randint(N)]]                   # first center: uniform random
        for _ in range(1, self.k):
            d = pairwise_sq_dist(X, np.array(centroids))        # (N, #chosen)
            nearest = d.min(axis=1)                             # (N,)
            total = nearest.sum()
            if total > 0:
                probs = nearest / total                         # (N,) sums to 1
            else:                                               # all points identical to a center
                probs = np.full(N, 1.0 / N)
            centroids.append(X[np.random.choice(N, p=probs)])
        return np.array(centroids)                              # (K, D)

    # ---- step 2: assignment ----
    def _assign(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (N,) index of nearest centroid."""
        d = pairwise_sq_dist(X, self.centroids)                 # (N, K)
        return d.argmin(axis=1)                                 # (N,)

    # ---- step 3: update ----
    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """X: (N, D), labels: (N,) -> new centroids (K, D). Empty cluster -> re-seed with a random point."""
        new = np.empty_like(self.centroids)                     # (K, D)
        for j in range(self.k):
            members = X[labels == j]                            # (n_j, D)
            if len(members) == 0:                               # empty cluster: mean of [] would be NaN
                new[j] = X[np.random.randint(len(X))]
            else:
                new[j] = members.mean(axis=0)
        return new

    def fit(self, X: np.ndarray) -> "KMeans":
        """X: (N, D) float. Returns self."""
        if self.init == "random":
            self.centroids = self._init_random(X)
        else:
            self.centroids = self._init_plusplus(X)
        for _ in range(self.max_iters):
            labels = self._assign(X)                            # (N,)
            new = self._update(X, labels)                       # (K, D)
            shift = np.linalg.norm(new - self.centroids)        # how far the centers moved
            self.centroids = new
            if shift < self.tol:                                # step 4: converged
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) float -> (N,) int cluster ids."""
        return self._assign(X)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



