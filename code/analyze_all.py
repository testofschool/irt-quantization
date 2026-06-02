#!/usr/bin/env python3
"""analyze_all.py — Full analysis: 2PL IRT, KL disturbance, DIF, bootstrap, CV.

Reads results/*.jsonl (per-subject response matrices) and computes all statistics,
writing them to findings.json (the single source of truth for the figures and paper).

Usage:
    python3 analyze_all.py          # full run -> findings.json
    python3 analyze_all.py --fast   # quick run -> findings_fast.json (no overwrite)

Probes (pre-specified):
    F1  accuracy floor by precision (easy vs challenge)
    F2  IRT difficulty b recovers ARC Easy/Challenge label
    F3  family invariance + half-subject stability of b
    F4  KL disturbance vs difficulty (within- and cross-family)
    F5  DIF shift Delta-b between precision groups (the core probe)
    F6  IRT vs simple averaging as difficulty estimator
"""
import json, os, sys, math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

FAST = "--fast" in sys.argv

PANEL = ["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_0", "Q4_K_M", "Q3_K_M", "Q2_K",
         "1.5B_Q8_0", "1.5B_Q4_K_M", "1.5B_Q2_K",
         "llama1B_Q8_0", "llama1B_Q4_K_M"]
EXTERNAL = "ext_3B"   # validator, never in panel

def load(s):
    path = f"results/{s}.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}

def softmax(x):
    x = np.asarray(x, float); x = x - x.max()
    e = np.exp(x); return e / e.sum()

def kl_item(D, base, q, ids):
    out = []
    for i in ids:
        p = softmax(D[base][i]["opt_lp"])
        r = softmax(D[q][i]["opt_lp"])
        out.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(r + 1e-12)))))
    return np.array(out)

def fit_2pl(X, l2=0.06, maxiter=350, mask=None):
    """Penalized 2PL MLE. X is (J subjects x K items) binary matrix.
    If mask is given (same shape, 1=observed, 0=missing), the likelihood is summed
    ONLY over observed cells -- the correct way to handle a sparse response matrix,
    rather than treating missing entries as incorrect.
    Returns (theta, b, a). b is centered."""
    J, K = X.shape
    W = np.ones_like(X) if mask is None else mask.astype(float)
    def nll(p):
        th = p[:J]; b = p[J:J+K]; a = np.exp(p[J+K:])
        z = a[None, :] * (th[:, None] - b[None, :])
        P = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
        ll = np.sum(W * (X * np.log(P) + (1 - X) * np.log(1 - P)))
        pen = l2 * (np.sum(th**2) + np.sum(b**2) / 4 + np.sum(p[J+K:]**2) / 0.5)
        return -(ll - pen)
    p0 = np.zeros(J + 2 * K)
    res = minimize(nll, p0, method="L-BFGS-B", options={"maxiter": maxiter})
    th = res.x[:J]; b = res.x[J:J+K]; a = np.exp(res.x[J+K:])
    return th, b - b.mean(), a

def boot_ci(x, y, n=5000, seed=7):
    rng = np.random.RandomState(seed); m = len(x); rs = []
    for _ in range(n if not FAST else 500):
        idx = rng.choice(m, m, True)
        rs.append(spearmanr(x[idx], y[idx]).correlation)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))

def main():
    avail = [s for s in PANEL if os.path.exists(f"results/{s}.jsonl")]
    if not avail:
        print("No results/*.jsonl found. Run run_eval.py first.")
        sys.exit(1)
    D = {s: load(s) for s in avail}
    ids = sorted(D[avail[0]].keys())
    strat = np.array([1 if D[avail[0]][i]["strat"] == "challenge" else 0 for i in ids])
    X = np.array([[D[s][i]["correct"] for i in ids] for s in avail], float)

    F = {}

    # F1: accuracy floor
    easy = [k for k in range(len(ids)) if strat[k] == 0]
    chal = [k for k in range(len(ids)) if strat[k] == 1]
    F["F1_accuracy_floor"] = {
        s: {"easy": float(np.mean([D[s][ids[k]]["correct"] for k in easy])),
            "chal": float(np.mean([D[s][ids[k]]["correct"] for k in chal]))}
        for s in avail}

    # F2: IRT b recovers ARC label
    th, b, a = fit_2pl(X)
    rho, p = spearmanr(b, strat)
    # AUC of b separating easy/chal
    order = np.argsort(b)
    ranks = np.empty_like(order); ranks[order] = np.arange(len(b))
    n1 = strat.sum(); n0 = len(strat) - n1
    auc = (ranks[strat == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
    F["F2_irt_recovery"] = {
        "spearman_b_vs_hard": round(float(rho), 3), "p_value": float(p),
        "auc": round(float(auc), 3),
        "mean_b_easy": round(float(b[strat == 0].mean()), 2),
        "mean_b_chal": round(float(b[strat == 1].mean()), 2)}
    np.save("results_b.npy", b)

    # F3: family invariance + half-subject stability
    qwen = [k for k, s in enumerate(avail) if not s.startswith("llama")]
    if len(qwen) >= 2:
        _, b_qwen, _ = fit_2pl(X[qwen])
        inv = spearmanr(b, b_qwen).correlation
    else:
        inv = float("nan")
    rng = np.random.RandomState(0); stab = []
    for _ in range(20 if not FAST else 5):
        half = rng.choice(len(avail), len(avail) // 2, False)
        if len(half) >= 2:
            _, bh, _ = fit_2pl(X[half]); stab.append(spearmanr(b, bh).correlation)
    F["F3_family_invariance"] = {
        "qwen_vs_full_rho": round(float(inv), 3),
        "half_subject_rho": round(float(np.mean(stab)), 3),
        "half_subject_std": round(float(np.std(stab)), 3)}

    # F4: KL vs difficulty (within- and cross-family). All pairs the paper reports.
    F["F4_kl_difficulty"] = {}
    pairs = [("0.5B_Q4_KM", "f16", "Q4_K_M"),
             ("1.5B_Q4_KM", "1.5B_Q8_0", "1.5B_Q4_K_M"),
             ("1.5B_Q2_K", "1.5B_Q8_0", "1.5B_Q2_K"),
             ("llama_Q4_KM", "llama1B_Q8_0", "llama1B_Q4_K_M")]
    for name, base, q in pairs:
        if base in D and q in D:
            kl = kl_item(D, base, q, ids)
            rr, pp = spearmanr(b, kl)
            lo, hi = boot_ci(b, kl)
            F["F4_kl_difficulty"][name] = {
                "rho": round(float(rr), 3), "p": round(float(pp), 4),
                "ci": [round(lo, 2), round(hi, 2)]}

    # F5: DIF shift (core probe)
    if EXTERNAL in [s.replace("results/", "") for s in os.listdir("results")] or \
       os.path.exists(f"results/{EXTERNAL}.jsonl"):
        D[EXTERNAL] = load(EXTERNAL)
        gt = -np.array([D[EXTERNAL][i]["lp_correct"] for i in ids])
        low = [k for k, s in enumerate(avail) if "Q2" in s or "Q3" in s]
        high = [k for k, s in enumerate(avail) if "Q8" in s or s == "f16"]
        if len(low) >= 2 and len(high) >= 2:
            _, b_low, _ = fit_2pl(X[low]); _, b_high, _ = fit_2pl(X[high])
            dB = b_low - b_high
            rr, _ = spearmanr(gt, dB); lo, hi = boot_ci(gt, dB)
            # 5-fold CV
            rng = np.random.RandomState(1); folds = np.array_split(rng.permutation(len(ids)), 5)
            cv = [spearmanr(gt[f], dB[f]).correlation for f in folds]
            F["F5_irt_dif"] = {
                "spearman_dB_vs_3Bdiff": round(float(rr), 3),
                "ci": [round(lo, 2), round(hi, 2)],
                "cv_folds": [round(float(c), 2) for c in cv]}

    # F6: IRT vs simple averaging.
    # Dense regime: full panel.
    avg = X.mean(0)
    rho_avg = spearmanr(avg, strat).correlation
    rho_irt = spearmanr(b, strat).correlation
    # Sparse regime: randomly mask 60% of (subject,item) cells (seeded), then compare
    # IRT difficulty vs item-mean accuracy on the masked matrix, averaged over repeats.
    rng_s = np.random.RandomState(20)
    reps = 5 if not FAST else 2
    d_sparse = []
    for _ in range(reps):
        M = (rng_s.rand(*X.shape) > 0.40).astype(float)  # observe ~60% of cells
        # IRT fit uses the masked likelihood (missing cells contribute nothing),
        # not zero-fill. Averaging uses only observed cells per item.
        _, b_s, _ = fit_2pl(X, mask=M)
        obs = M.sum(0)
        col_mean = np.where(obs > 0, (X * M).sum(0) / np.maximum(obs, 1), float(np.mean(X)))
        r_irt_s = abs(spearmanr(b_s, strat).correlation)
        r_avg_s = abs(spearmanr(col_mean, strat).correlation)
        d_sparse.append(r_irt_s - r_avg_s)
    F["F6_irt_vs_averaging"] = {
        "delta_rho_dense": round(float(abs(rho_irt) - abs(rho_avg)), 3),
        "delta_rho_sparse": round(float(np.mean(d_sparse)), 3),
        "sparse_keep_frac": 0.60, "sparse_reps": reps, "sparse_seed": 20,
        "sparse_method": "masked-likelihood IRT (observed cells only)"}

    out = "findings_fast.json" if FAST else "findings.json"
    json.dump(F, open(out, "w"), indent=2)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
