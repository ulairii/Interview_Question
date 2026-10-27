"""Regenerate the collapsible "标准答案" block in every note from code/ml_coding.py.

Run from the repo root:  python tools/sync_answers.py
"""
import importlib, inspect, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
M = importlib.import_module("ml_coding")

# note path (under notes/)  ->  objects of ml_coding shown as the answer ("@..." = verbatim line)
ANSWERS = {
    "losses_metrics/01_Chamfer_Distance": ["pairwise_sq_dist", "chamfer_distance"],
    "classic_ml/01_KNN": ["pairwise_sq_dist", "KNN"],
    "classic_ml/02_KMeans": ["pairwise_sq_dist", "KMeans"],
    "classic_ml/03_线性与逻辑回归": ["sigmoid", "LogisticRegression"],
    "classic_ml/04_决策树与随机森林": ["_Node", "DecisionTree", "RandomForest"],
    "classic_ml/05_GMM": ["GMM"],
    "classic_ml/06_PCA": ["PCA", "pca"],
    "dl_components/attention/01_SDPA与Mask": ["causal_mask", "scaled_dot_product_attention"],
    "dl_components/attention/02_SelfAttention": ["attention", "self_attention"],
    "dl_components/attention/03_CrossAttention": ["cross_attention"],
    "dl_components/attention/04_MultiHead": ["multi_head_attention"],
    "dl_components/attention/05_KVCache": ["@KVCache = Tuple[torch.Tensor, torch.Tensor]   # (k, v), each (B, H, S, dk)",
                                           "MultiHeadAttention"],
    "dl_components/01_BN_LN_RMSNorm": ["BatchNorm1d", "LayerNorm", "RMSNorm"],
    "dl_components/02_Conv2d_MaxPool": ["_windows", "conv2d", "maxpool2d"],
    "dl_components/03_Dropout_LoRA": ["InvertedDropout", "LoRALinear"],
    "losses_metrics/02_损失函数": ["stable_cross_entropy", "binary_focal_loss", "infonce_loss"],
    "losses_metrics/03_AUC": ["auc_roc", "auc_rank"],
    "losses_metrics/04_IoU_GIoU_NMS": ["box_iou_giou", "nms"],
    "sampling/01_采样与解码": ["softmax", "top_k_filter", "top_p_filter", "sample_top_k_top_p"],
}
START, END = "<!-- ANSWER-START -->", "<!-- ANSWER-END -->"
NOTE = "\n---\n\n## ✏️ 我的笔记"
HEAD = "from typing import List, Optional, Tuple\n\nimport numpy as np\n"
TORCH = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
DOWN = lambda f: "../" * (len(f.relative_to(ROOT / "notes").parts))


def build(names):
    parts = [n[1:] if n.startswith("@") else inspect.getsource(getattr(M, n)).rstrip() for n in names]
    body = "\n\n\n".join(parts)
    head = HEAD + (TORCH if re.search(r"torch|nn\.|F\.", body) else "")
    return head + "\n\n" + body


for stem, names in ANSWERS.items():
    f = ROOT / "notes" / f"{stem}.md"
    t = f.read_text()
    t = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", "", t, flags=re.S)
    assert NOTE in t, f
    block = f"""{START}
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py]({DOWN(f)}code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
{build(names)}
```

</details>
{END}
"""
    f.write_text(t.replace(NOTE, "\n" + block + NOTE, 1))
    print("synced", stem)
