"""练习/模拟面试入口。用法见 exercises.py 顶部。"""
import importlib, os, random, shutil, sys, time
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
torch.manual_seed(0)
rng = np.random.default_rng(0)


def close(a, b, tol=1e-5, msg=""):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{msg} shape {a.shape} != {b.shape}"
    assert np.allclose(a, b, atol=tol), f"{msg} values differ:\n{a}\nvs\n{b}"


def blobs():
    X = np.r_[rng.normal(0, .3, (50, 2)), rng.normal(3, .3, (50, 2))]
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    Xt = np.r_[rng.normal(0, .3, (20, 2)), rng.normal(3, .3, (20, 2))]
    yt = np.r_[np.zeros(20), np.ones(20)].astype(int)
    return X, y, Xt, yt


# ------------------------------------------------------------------ checks
def t_chamfer(m):
    P, Q = rng.random((30, 3)), rng.random((40, 3))
    d = ((P[:, None] - Q[None]) ** 2).sum(-1)
    close(m.chamfer_distance(P, Q), d.min(1).mean() + d.min(0).mean(), msg="squared")
    close(m.chamfer_distance(P, Q, squared=False), np.sqrt(d).min(1).mean() + np.sqrt(d).min(0).mean(), msg="non-squared")
    assert abs(m.chamfer_distance(P, P)) < 1e-6, "CD(P,P) should be 0 (clamp negative fp values!)"
    assert not np.isnan(m.chamfer_distance(P, P, squared=False)), "NaN: sqrt of a tiny negative number"
    assert isinstance(m.chamfer_distance(P, Q), float), "return a python float"


def t_knn(m):
    X, y, Xt, yt = blobs()
    k = m.KNN(5); k.fit(X, y)
    assert (k.predict(Xt) == yt).all()
    k = m.KNN(500); k.fit(X, y); k.predict(Xt)      # k > N must not crash


def _make_km(m, k, init, seed):
    """Works whether or not the student's KMeans takes a seed (np.random.seed is set as a fallback)."""
    import inspect
    np.random.seed(seed)
    kw = dict(init=init)
    params = inspect.signature(m.KMeans.__init__).parameters
    if "seed" in params:
        kw["seed"] = seed
    kk = "n_clusters" if "n_clusters" in params else next(n for n in params if n not in ("self", "init", "seed", "max_iters", "tol"))
    return m.KMeans(**{kk: k}, **kw)


def t_kmeans(m):
    X, y, *_ = blobs()
    for init in ("random", "kmeans++"):
        km = _make_km(m, 2, init, 0).fit(X)
        lab = km.predict(X)
        assert (lab == y).all() or (lab == 1 - y).all(), f"init={init}: two blobs should be separated"
        assert km.centroids.shape == (2, 2)
    Xd = np.r_[np.zeros((20, 2)), np.ones((5, 2)) * 5]       # duplicate points: empty cluster / zero-prob risk
    for init in ("random", "kmeans++"):
        km = _make_km(m, 4, init, 1).fit(Xd)
        assert not np.isnan(km.centroids).any(), f"init={init}: NaN centroids, empty cluster not handled"


def t_logreg(m):
    X, y, Xt, yt = blobs()
    lr = m.LogisticRegression(lr=0.5, epochs=500).fit(X, y)
    assert ((lr.predict_proba(Xt) > .5) == yt).all()
    big = m.LogisticRegression(lr=0.5, epochs=5).fit(X * 1e4, y)
    assert np.isfinite(big.predict_proba(X * 1e4)).all(), "overflow with large inputs"


def t_tree(m):
    X, y, Xt, yt = blobs()
    assert (m.DecisionTree(3).fit(X, y).predict(Xt) == yt).all()
    Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, float); yx = np.array([0, 1, 1, 0] * 5)
    assert (m.DecisionTree(3).fit(Xx, yx).predict(Xx) == yx).all(), "XOR needs depth 2"


def t_pca(m):
    X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
    Z = m.pca(X, 2)
    assert Z.shape == (200, 2)
    close(Z.var(0, ddof=1), np.linalg.eigvalsh(np.cov(X.T))[::-1][:2], 1e-6, "variance of components")
    close(Z.mean(0), np.zeros(2), 1e-8, "must be centred")


def t_ce(m):
    lg, tg = rng.normal(size=(10, 5)) * 50, rng.integers(0, 5, 10)
    close(m.stable_cross_entropy(lg, tg), F.cross_entropy(torch.tensor(lg), torch.tensor(tg)).item())
    big = np.array([[1000., -1000.], [-1000., 1000.]])
    assert np.isfinite(m.stable_cross_entropy(big, np.array([0, 1]))), "overflow"


def t_focal_infonce(m):
    lb, tb = torch.randn(32), torch.randint(0, 2, (32,))
    close(m.binary_focal_loss(lb, tb, 0.5, 0.0).item(), 0.5 * F.binary_cross_entropy_with_logits(lb, tb.float()).item(), msg="gamma=0 alpha=.5 -> BCE/2")
    assert np.isfinite(m.binary_focal_loss(torch.tensor([1e4, -1e4]), torch.tensor([0, 1]))).all()
    z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
    ref = F.cross_entropy(F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T / 0.07, torch.arange(8))
    close(m.infonce_loss(z1, z2).item(), ref.item(), msg="infonce")


def t_auc(m):
    yt, ys = rng.integers(0, 2, 200), rng.random(200)
    for sc in (ys, np.round(ys, 1)):
        brute = np.mean([(a > b) + .5 * (a == b) for a, b in product(sc[yt == 1], sc[yt == 0])])
        close(m.auc_roc(yt, sc), brute, msg="vs brute force (ties!)")
    assert m.auc_roc(np.ones(5, int), rng.random(5)) == 0.5, "single class -> 0.5"


def _boxes(n):
    a = rng.random((n, 2)); wh = rng.random((n, 2)) * .5 + .1
    return np.c_[a, a + wh]


def t_iou(m):
    b1, b2 = _boxes(5), _boxes(7)
    iou, giou = m.box_iou_giou(b1, b2)
    assert iou.shape == giou.shape == (5, 7)
    ri, rg = np.zeros((5, 7)), np.zeros((5, 7))
    for i, a in enumerate(b1):
        for j, b in enumerate(b2):
            iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            ri[i, j] = inter / u
            ea = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))
            rg[i, j] = ri[i, j] - (ea - u) / ea
    close(iou, ri, msg="iou"); close(giou, rg, msg="giou")


def t_nms(m):
    b = np.array([[0, 0, 1, 1], [0, 0, 1, .95], [2, 2, 3, 3], [0.05, 0, 1, 1]], float)
    s = np.array([.9, .8, .7, .95])
    assert m.nms(b, s, .5) == [3, 2], m.nms(b, s, .5)
    bb, ss = _boxes(30), rng.random(30)
    keep = m.nms(bb, ss, .3)
    iou, _ = m.box_iou_giou(bb[keep], bb[keep]) if hasattr(m, "box_iou_giou") else (None, None)
    assert keep[0] == int(ss.argmax()) and len(set(keep)) == len(keep)


def t_attention(m):
    B, T, S, C, H = 2, 6, 9, 32, 4
    q, k, v = (torch.randn(B, H, n, 8) for n in (T, S, S))
    out, w = m.scaled_dot_product_attention(q, k, v)
    close(out, F.scaled_dot_product_attention(q, k, v), msg="sdpa"); close(w.sum(-1), np.ones((B, H, T)), msg="softmax over keys")
    cm = m.causal_mask(T, T); assert cm.dtype == torch.bool and bool(cm[0, 1]) is False and bool(cm[1, 0]) is True
    qc, kc, vc = (torch.randn(B, H, T, 8) for _ in range(3))
    close(m.scaled_dot_product_attention(qc, kc, vc, cm)[0], F.scaled_dot_product_attention(qc, kc, vc, is_causal=True), msg="causal")
    assert bool(m.causal_mask(1, 5).all()), "decoding step T=1 sees every key"
    x, ctx = torch.randn(B, T, C), torch.randn(B, S, C)
    Wq, Wk, Wv = (torch.randn(C, 8) for _ in range(3))
    close(m.self_attention(x, Wq, Wk, Wv, causal=True), F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, is_causal=True), msg="self")
    pad = torch.ones(B, S, dtype=torch.bool); pad[:, -3:] = False
    ref = F.scaled_dot_product_attention(x @ Wq, ctx @ Wk, ctx @ Wv, attn_mask=pad[:, None, :])
    close(m.cross_attention(x, ctx, Wq, Wk, Wv, pad), ref, msg="cross with padding")
    ctx2 = ctx.clone(); ctx2[:, -3:] += 100
    close(m.cross_attention(x, ctx2, Wq, Wk, Wv, pad), ref, msg="padding must be ignored")


def t_mha(m):
    import ml_coding as r
    B, T, S, C, H = 2, 6, 9, 32, 4
    x, ctx = torch.randn(B, T, C), torch.randn(B, S, C)
    W = [torch.randn(C, C) / C ** .5 for _ in range(4)]
    cm = r.causal_mask(T, T)
    close(m.multi_head_attention(x, x, *W, H, cm), r.multi_head_attention(x, x, *W, H, cm), msg="self+causal")
    close(m.multi_head_attention(x, ctx, *W, H), r.multi_head_attention(x, ctx, *W, H), msg="cross T!=S")


def t_norm(m):
    xb = torch.randn(16, 8)
    bn, ref = m.BatchNorm1d(8), torch.nn.BatchNorm1d(8)
    close(bn(xb).detach(), ref(xb).detach(), 1e-5, "bn train")
    assert "running_mean" in dict(bn.named_buffers()), "running stats must be buffers"
    assert len(list(bn.parameters())) == 2
    xb2 = torch.randn(16, 8)
    bn(xb2); ref(xb2)
    bn.eval(); ref.eval()
    close(bn(xb).detach(), ref(xb).detach(), 1e-5, "bn eval (running stats / unbiased var)")
    close(m.LayerNorm(8)(xb).detach(), torch.nn.LayerNorm(8)(xb).detach(), 1e-5, "layernorm")


def t_dropout(m):
    d = m.InvertedDropout(0.3); x = torch.ones(200000)
    assert abs(d(x).mean().item() - 1) < 0.01, "expectation must be preserved (scale by 1/(1-p))"
    assert 0.25 < (d(x) == 0).float().mean().item() < 0.35
    d.eval(); assert (d(x) == x).all(), "eval = identity"


def t_lora(m):
    lo = m.LoRALinear(16, 8); x = torch.randn(4, 16)
    close(lo(x).detach(), lo.base(x).detach(), 1e-6, "delta W must be 0 at init (B = zeros)")
    assert not lo.base.weight.requires_grad and lo.A.requires_grad and lo.B.requires_grad
    assert (lo.A.abs().sum() > 0), "A must be random, not zeros"
    lo.B.data.normal_()
    ref = lo.base(x) + (x @ lo.A.T @ lo.B.T) * (16.0 / 8)
    close(lo(x).detach(), ref.detach(), 1e-5, "scale = alpha / r")


def t_conv(m):
    x, w, b = rng.normal(size=(2, 3, 9, 9)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    for s, p in [(1, 0), (2, 1)]:
        ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=s, padding=p).numpy()
        close(m.conv2d(x, w, b, s, p), ref, 1e-8, f"stride={s} pad={p}")
    close(m.conv2d(x, w), F.conv2d(torch.tensor(x), torch.tensor(w)).numpy(), 1e-8, "no bias")


def _draw(m, seed, **kw):
    """Call the student's sampler; pass rng only if it takes one, else seed np.random."""
    import inspect
    np.random.seed(seed)
    if "rng" in inspect.signature(m.sample_top_k_top_p).parameters:
        kw["rng"] = np.random.default_rng(seed)
    return m.sample_top_k_top_p(**kw)


def t_sampling(m):
    lg = np.array([5., 4., 0., -1.])
    assert {_draw(m, i, logits=lg, k=2) for i in range(200)} <= {0, 1}, "top-k"
    assert {_draw(m, i, logits=lg, p=0.5) for i in range(200)} == {0}, "top-p must keep the top token"
    c = np.bincount([_draw(m, i, logits=lg, temperature=100) for i in range(2000)], minlength=4)
    assert c.min() > 300, "high temperature should be near uniform"
    assert _draw(m, 0, logits=np.array([1000., 0.])) == 0, "overflow-safe softmax"


# name: (title, minutes, check, prompt, follow-ups)
PROBLEMS = {
    "chamfer": ("Chamfer Distance", 10, t_chamfer, "实现两个点云 P:(N,D)、Q:(M,D) 的 Chamfer Distance，要求无 for 循环。", ["为什么要双向？", "为什么 mean 不是 sum？", "N=M=1e5 内存怎么办？", "广播版 vs matmul 展开，数值上要注意什么？", "密度不均怎么办（EMD）？"]),
    "knn": ("KNN 分类器", 10, t_knn, "写一个 KNN 分类器（fit/predict），距离矩阵无循环。", ["复杂度？", "argpartition 怎么用，kth 填几？", "平票怎么办？", "特征尺度？", "k > N 呢？"]),
    "kmeans": ("K-Means → K-Means++", 15, t_kmeans, "先写普通 K-Means（随机初始化），写完再加 K-Means++ 初始化。先写骨架再填。", ["空簇怎么处理？", "为什么一定收敛？局部最优怎么办？", "怎么选 K？", "N=1e5, K=100, D=512 的内存？", "和 GMM 的关系？"]),
    "logreg": ("Logistic Regression", 12, t_logreg, "手写二分类逻辑回归（梯度下降 + L2），口述梯度推导。", ["为什么梯度是 p-y？", "sigmoid 溢出怎么办？", "bias 要正则吗？", "扩展到多分类？", "和线性回归梯度的关系？"]),
    "tree": ("决策树 (CART)", 20, t_tree, "写 CART 分类树：Gini、最佳分裂、递归建树、预测。", ["Gini vs 熵？", "怎么改成随机森林？", "复杂度怎么优化（排序+累计）？", "防过拟合？", "回归树怎么改？"]),
    "pca": ("PCA", 8, t_pca, "用 SVD 实现 PCA，返回投影后的数据。", ["为什么 SVD 比 eig(XᵀX) 好？", "解释方差比怎么算？", "要不要标准化？", "D≫N 时？"]),
    "ce": ("Cross-Entropy", 8, t_ce, "实现数值稳定的 softmax 交叉熵（logits 可能很大）。", ["为什么不能先 softmax 再 log？", "梯度是什么？", "label smoothing 怎么加？"]),
    "focal_infonce": ("Focal Loss + InfoNCE", 15, t_focal_infonce, "实现二分类 focal loss 与 InfoNCE。", ["gamma 的作用？", "alpha 的作用？", "温度 τ 的作用？", "InfoNCE 为什么是交叉熵？", "怎么做对称版？"]),
    "auc": ("AUC-ROC", 12, t_auc, "实现 AUC，要求 O(N log N) 并处理分数并列。", ["物理意义？", "秩和法为什么并列要平均秩？", "PR-AUC 什么时候更合适？", "只有一类时？"]),
    "iou": ("IoU / GIoU", 12, t_iou, "向量化计算 (N,4) 与 (M,4) 框的两两 IoU 和 GIoU。", ["GIoU 比 IoU 好在哪？", "怎么扩展到 3D？", "旋转框怎么办？", "除零保护？"]),
    "nms": ("NMS", 10, t_nms, "实现 NMS（可复用你的 IoU）。", ["复杂度？", "多类别怎么做？", "Soft-NMS？", "阈值怎么选？"]),
    "attention": ("Attention 三件套", 25, t_attention, "写 causal_mask、scaled_dot_product_attention，再用它写 self_attention 和 cross_attention。", ["为什么除 √dk？", "T=1 解码时 mask 是什么？", "cross 为什么不要 causal？", "padding mask 怎么加？", "全 -inf 行会怎样？"]),
    "mha": ("Multi-Head Attention", 20, t_mha, "写函数版多头注意力（self 与 cross 通用），每行注释 shape。", ["view 与 transpose 的顺序？", "为什么要 contiguous？", "参数量？", "MQA/GQA？", "KV cache 怎么加？"]),
    "norm": ("BatchNorm + LayerNorm", 18, t_norm, "手写 BatchNorm1d（含 running 统计）和 LayerNorm。", ["train/eval 区别？", "running_var 为什么是 buffer？", "为什么 LLM 用 RMSNorm？", "BN 在小 batch 的问题？"]),
    "dropout": ("Inverted Dropout", 6, t_dropout, "实现 Dropout 模块。", ["为什么缩放放在训练阶段？", "eval 时？", "p=1 的边界？"]),
    "lora": ("LoRA Layer", 12, t_lora, "实现 LoRALinear：冻结原权重，加低秩更新。", ["A、B 怎么初始化，为什么？", "怎么合并权重？", "参数量？", "scale 为什么是 alpha/r？"]),
    "conv": ("Conv2d", 15, t_conv, "用 NumPy 实现 conv2d（stride、padding）。可先写循环版再向量化。", ["输出尺寸公式？", "im2col 原理？", "和互相关的区别？", "复杂度？"]),
    "sampling": ("Top-k / Top-p 采样", 12, t_sampling, "实现带 temperature 的 top-k / top-p 采样。", ["temperature 的作用？", "top-p 为什么保证至少一个 token？", "和 beam search 区别？", "数值稳定？"]),
}


def get_module(use_ref):
    return importlib.import_module("ml_coding") if use_ref else importlib.reload(importlib.import_module("exercises")) if "exercises" in sys.modules else importlib.import_module("exercises")


def run_check(name, m):
    title, _, fn, *_ = PROBLEMS[name]
    try:
        fn(m)
        print(f"  PASS  {name:14s} {title}")
        return True
    except NotImplementedError:
        print(f"  TODO  {name:14s} {title}")
    except Exception as e:
        print(f"  FAIL  {name:14s} {title}\n        {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
    return False


def main():
    args = sys.argv[1:]
    use_ref = "--ref" in args
    args = [a for a in args if a != "--ref"]
    cmd = args[0] if args else "list"
    if cmd == "list":
        for n, (t, mins, *_ ) in PROBLEMS.items():
            print(f"{n:14s} {mins:3d} min  {t}")
    elif cmd == "check":
        names = list(PROBLEMS) if len(args) < 2 or args[1] == "all" else args[1:]
        m = get_module(use_ref)
        ok = [run_check(n, m) for n in names]
        print(f"\n{sum(ok)}/{len(ok)} passed")
    elif cmd == "reset":
        shutil.copy(os.path.join(HERE, "template", "exercises_blank.py"), os.path.join(HERE, "exercises.py"))
        print("exercises.py reset to blank (all problems).")
    elif cmd == "mock":
        name = args[1] if len(args) > 1 else random.choice(list(PROBLEMS))
        title, mins, fn, prompt, fups = PROBLEMS[name]
        print(f"\n=== 模拟面试：{title}（限时 {mins} 分钟）===\n{prompt}\n")
        print(f"在 practice/exercises.py 里作答。先口述思路/边界条件，再写。完成后按 Enter 交卷。")
        t0 = time.time(); input()
        used = (time.time() - t0) / 60
        print(f"\n用时 {used:.1f} 分钟（限时 {mins}）{'  ⚠ 超时' if used > mins else ''}")
        m = get_module(use_ref)
        run_check(name, m)
        print("\n追问（自己口头回答，再对照 notes/）：")
        for i, q in enumerate(fups, 1):
            print(f"  {i}. {q}")
        print("\n自评：规范(类型/shape注释)? 边界条件? 复杂度? 优化?")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
