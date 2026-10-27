# KNN（面试精简版）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


```python
def pairwise_sq_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:    # 和 Chamfer / K-Means 共用
    p_sq = np.sum(P ** 2, axis=1)                  # (M,)
    q_sq = np.sum(Q ** 2, axis=1)                  # (N,)
    cross = P @ Q.T                                # (M, N)
    d = p_sq[:, None] + q_sq[None, :] - 2 * cross  # (M, N)
    return np.maximum(d, 0.0)


class KNN:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X, y                                  # (N,D), (N,)

    def predict(self, Xt: np.ndarray) -> np.ndarray:           # Xt:(M,D) -> (M,) int
        d = pairwise_sq_dist(Xt, self.X)                       # (M,N)
        idx = np.argsort(d, axis=1)[:, :self.k]                # (M,k)
        labels = self.y[idx]                                   # (M,k)
        return np.array([np.bincount(r).argmax() for r in labels])
```

**为什么用这套（和 Chamfer 同款）**：距离矩阵是 KNN 的瓶颈，matmul 展开只需 `(M,N)` 内存且走 BLAS。
若一时想不起展开式，可先写**广播版**（正确、好记，只是内存是 `(M,N,D)`），再说"可以优化成 matmul"：

```python
d = np.sum((Xt[:, None, :] - self.X[None, :, :]) ** 2, axis=-1)   # (M,N)
```

四步：**距离矩阵 → 取 k 个最近 → 取标签 → 多数投票**。

追问清单：

| 追问 | 回答 |
|---|---|
| 内存太大？ | 用 `||x||²+||c||²-2xc^T` 展开，`(M,N)` 而非 `(M,N,D)`；测试集再分块 |
| 更快取前 k？ | `np.argpartition(d, k-1, axis=1)[:, :k]`，O(N)；**kth 是 `k-1` 不是 `k`**（`kth=k` 在 k≥N 时越界；k>N 需先 `min(k,N)`） |
| 投票平票？ | `bincount.argmax` 取最小类别编号；可改距离加权 `w=1/(d+eps)` |
| 标签不是 0..C-1 整数？ | 先 `np.unique(return_inverse=True)` 映射；`bincount` 只接受非负整数 |
| 回归？ | `self.y[idx].mean(axis=1)` |
| 特征尺度？ | 先标准化，否则大尺度特征主导距离 |
| 复杂度？ | 预测 O(M·N·D)；大数据用 KD-tree / ball-tree / FAISS（高维 KD-tree 退化） |
| 为什么平方距离可直接比较？ | 平方单调，排序不变，省 sqrt |

`KNNFast`（matmul + argpartition，k>N 安全）在 `code/ml_coding.py`。


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


class KNN:
    """Interview-length KNN classifier. Labels must be non-negative ints."""

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, D) float, y: (N,) int in [0, C)."""
        self.X, self.y = X, y

    def predict(self, Xt: np.ndarray) -> np.ndarray:
        """Xt: (M, D) float -> (M,) int predicted labels."""
        d = pairwise_sq_dist(Xt, self.X)                        # (M, N)
        idx = np.argsort(d, axis=1)[:, :self.k]                 # (M, k)
        labels = self.y[idx]                                    # (M, k)
        return np.array([np.bincount(r).argmax() for r in labels])
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



