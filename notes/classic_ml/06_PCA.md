# PCA

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


```python
Xc = X - X.mean(0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)    # Xc = U S Vt
Z = Xc @ Vt[:n].T                                    # (N, n)
```

- 为什么用 SVD 而非 `eig(XᵀX)`：显式协方差使条件数平方，丢精度；D≫N 时 `XᵀX` 是 D×D，经济型 SVD 只与 `min(N,D)` 相关。
- 关系：协方差特征值 `λᵢ = sᵢ²/(N-1)`；解释方差比 `sᵢ²/Σsⱼ²`。
- 必须中心化；特征尺度差别大时先标准化。
- 主成分符号不唯一（±），测试时比较方差而非坐标。

<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np


class PCA:
    def __init__(self, n_components: int = 2) -> None:
        self.n = n_components
        self.mean: np.ndarray                                   # (D,)
        self.components: np.ndarray                             # (D, n)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) float -> (N, n) projected data."""
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)       # Xc = U S Vt
        self.components = Vt[:self.n].T                         # (D, n)
        return Xc @ self.components


def pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """X: (N, D) float -> (N, n_components) projection onto the top principal components."""
    return PCA(n_components).fit_transform(X)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



