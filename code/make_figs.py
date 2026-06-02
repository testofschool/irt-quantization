#!/usr/bin/env python3
"""Generate publication figures for the IRT-Quantization paper.

BPW values from the official llama.cpp quantize README (Llama-3.1-8B reference);
actual BPW varies slightly by architecture (K-quant presets mix precision).

TWO MODES, selected automatically by USE_RAW:
  * USE_RAW = True  (full results/*.jsonl panel present): figures are generated from
    measured response matrices. Panel-level summary values are read from findings.json,
    which should be produced by analyze_all.py immediately before figure generation.
  * USE_RAW = False (no matrices): figures are rendered as clearly labeled illustrative
    schematics using example_findings.json. Those values demonstrate output format only;
    they are placeholders, not reported numbers or measurements.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
results_dir = os.path.join(root_dir, "results")
os.makedirs(os.path.join(root_dir, "figures"), exist_ok=True)

PANEL = ["f16","Q8_0","Q6_K","Q5_K_M","Q4_0","Q4_K_M","Q3_K_M","Q2_K",
         "1.5B_Q8_0","1.5B_Q4_K_M","1.5B_Q2_K","llama1B_Q8_0","llama1B_Q4_K_M"]

def _raw_ready():
    """Raw mode requires the full panel present. Partial dirs -> reconstruction."""
    missing = [s for s in PANEL if not os.path.exists(os.path.join(results_dir, f"{s}.jsonl"))]
    if missing:
        print(f"make_figs: {len(missing)} panel file(s) missing "
              f"(e.g. {missing[0]}.jsonl) -- staying in reconstruction mode")
        return False
    return True

USE_RAW = _raw_ready()
findings_path = os.path.join(root_dir, "findings.json")
example_path = os.path.join(root_dir, "example_findings.json")
if USE_RAW:
    if not os.path.exists(findings_path):
        raise RuntimeError(
            "Raw response matrices are present, but findings.json is missing. "
            "Run `python code/analyze_all.py` before `python code/make_figs.py` so "
            "all panels use measured statistics rather than placeholders."
        )
    F = json.load(open(findings_path))
else:
    F = json.load(open(example_path))

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.linewidth": 0.8
})

BPW = {
    "f16": 16.00, "Q8_0": 8.50, "Q6_K": 6.56, "Q5_K_M": 5.70,
    "Q4_0": 4.67, "Q4_K_M": 4.89, "Q3_K_M": 4.00, "Q2_K": 3.16,
    "1.5B_Q8_0": 8.50, "1.5B_Q4_K_M": 4.89, "1.5B_Q2_K": 3.16,
    "llama1B_Q8_0": 8.50, "llama1B_Q4_K_M": 4.89,
}

# ----------------------------------------------------------------------------
# Real-data helpers (only exercised when USE_RAW)
# ----------------------------------------------------------------------------
def _load(subject):
    path = os.path.join(results_dir, f"{subject}.jsonl")
    rows = [json.loads(l) for l in open(path)]
    return {r["id"]: r for r in rows}

def _softmax(x):
    x = np.asarray(x, float); x = x - x.max()
    e = np.exp(x); return e / e.sum()

def _fit_2pl(X, l2=0.06, maxiter=300):
    """Penalized 2PL MLE. X is (J subjects x K items) binary. Returns centered b."""
    from scipy.optimize import minimize
    J, K = X.shape
    def nll(p):
        th = p[:J]; b = p[J:J+K]; a = np.exp(p[J+K:])
        z = a[None, :] * (th[:, None] - b[None, :])
        P = np.clip(1/(1+np.exp(-z)), 1e-6, 1-1e-6)
        ll = np.sum(X*np.log(P) + (1-X)*np.log(1-P))
        pen = l2*(np.sum(th**2) + np.sum(b**2)/4 + np.sum(p[J+K:]**2)/0.5)
        return -(ll - pen)
    res = minimize(nll, np.zeros(J+2*K), method="L-BFGS-B", options={"maxiter": maxiter})
    b = res.x[J:J+K]
    return b - b.mean()

def _kl_per_item(D, base, q, ids):
    out = []
    for i in ids:
        p = _softmax(D[base][i]["opt_lp"]); r = _softmax(D[q][i]["opt_lp"])
        out.append(float(np.sum(p*(np.log(p+1e-12) - np.log(r+1e-12)))))
    return np.array(out)

raw = {}
if USE_RAW:
    avail = [s for s in PANEL if os.path.exists(os.path.join(results_dir, f"{s}.jsonl"))]
    D = {s: _load(s) for s in avail}
    ids = sorted(D[avail[0]].keys())
    strat = np.array([1 if D[avail[0]][i]["strat"] == "challenge" else 0 for i in ids])
    X = np.array([[D[s][i]["correct"] for i in ids] for s in avail], float)
    b = _fit_2pl(X)
    raw = {"D": D, "ids": ids, "strat": strat, "b": b, "avail": avail}
    print("make_figs: USE_RAW=True -- plotting real per-item data")
else:
    print("make_figs: USE_RAW=False -- reconstruction mode from example_findings.json")
    rng = np.random.RandomState(42)

# ============================ FIGURE 1 ======================================
fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4), constrained_layout=True)
if USE_RAW:
    fig.suptitle("MEASURED PROBE OUTPUTS — generated from supplied response matrices",
                 fontsize=11, fontweight="bold")
else:
    fig.suptitle("ILLUSTRATIVE SCHEMATIC — not measured data (probe output layouts)",
                 fontsize=11, fontweight="bold", color="#a00000")

# --- A: Accuracy floor (always from example_findings.json summary) ---
ax = axs[0, 0]
fl = F["F1_accuracy_floor"]
fams = [
    ("Qwen-0.5B", ["f16","Q8_0","Q4_K_M","Q3_K_M","Q2_K"], "o", "#1f77b4"),
    ("Qwen-1.5B", ["1.5B_Q8_0","1.5B_Q4_K_M","1.5B_Q2_K"], "s", "#ff7f0e"),
    ("Llama-1B",  ["llama1B_Q8_0","llama1B_Q4_K_M"], "^", "#2ca02c"),
]
for name, subs, mk, c in fams:
    xs = [BPW[s] for s in subs]
    ax.plot(xs, [fl[s]["easy"] for s in subs], mk+"-", color=c, ms=5, lw=1.3, label=f"{name} easy")
    ax.plot(xs, [fl[s]["chal"] for s in subs], mk+"--", color=c, ms=5, lw=1.3, mfc="white", label=f"{name} chal")
ax.axhline(0.25, ls=":", c="gray", lw=1)
ax.text(15.5, 0.265, "chance (0.25)", fontsize=6.5, color="gray", va="bottom")
ax.set_xlabel("effective bits / weight"); ax.set_ylabel("accuracy")
ax.set_ylim(0.15, 1.0); ax.set_title("A  Probe 2: accuracy-floor masking")
ax.invert_xaxis(); ax.legend(ncol=2, fontsize=6.3, loc="upper center", framealpha=0.9)

# --- B: IRT difficulty b recovers ARC label ---
ax = axs[0, 1]
if USE_RAW:
    b = raw["b"]; strat = raw["strat"]
    b_easy = b[strat == 0]; b_chal = b[strat == 1]
else:
    b_easy = rng.normal(F["F2_irt_recovery"]["mean_b_easy"], 0.8, 50)
    b_chal = rng.normal(F["F2_irt_recovery"]["mean_b_chal"], 0.9, 50)
ax.hist(b_easy, bins=15, alpha=0.6, color="tab:green", label="ARC-Easy items")
ax.hist(b_chal, bins=15, alpha=0.6, color="tab:red", label="ARC-Challenge items")
rho = F["F2_irt_recovery"]["spearman_b_vs_hard"]; p = F["F2_irt_recovery"]["p_value"]; auc = F["F2_irt_recovery"]["auc"]
ax.set_title("B  Probe 1: IRT difficulty b vs ARC label\n(decision stat: Spearman / AUC)")
ax.set_xlabel("IRT-estimated item difficulty b"); ax.set_ylabel("count"); ax.legend(fontsize=8)

# --- C: KL disturbance vs b (1.5B family) ---
ax = axs[1, 0]
if USE_RAW:
    b = raw["b"]; D = raw["D"]; ids = raw["ids"]
    series = []
    if "1.5B_Q8_0" in D and "1.5B_Q4_K_M" in D:
        series.append(("1.5B Q4", "tab:blue", _kl_per_item(D, "1.5B_Q8_0", "1.5B_Q4_K_M", ids)))
    if "1.5B_Q8_0" in D and "1.5B_Q2_K" in D:
        series.append(("1.5B Q2", "tab:purple", _kl_per_item(D, "1.5B_Q8_0", "1.5B_Q2_K", ids)))
    for q, c_color, kl in series:
        from scipy.stats import spearmanr
        rr = spearmanr(b, kl).correlation
        ax.scatter(b, kl, s=14, alpha=0.5, color=c_color, label=f"{q} (\u03c1={rr:+.2f})")
        z = np.polyfit(b, kl, 1); xx = np.linspace(b.min(), b.max(), 50)
        ax.plot(xx, np.polyval(z, xx), color=c_color, lw=1.5)
else:
    b_all = np.concatenate([b_easy, b_chal])
    for q, c_color, rho_val in [
        ("1.5B Q4", "tab:blue", F["F4_kl_difficulty"]["1.5B_Q4_KM"]["rho"]),
        ("1.5B Q2", "tab:purple", F["F4_kl_difficulty"]["1.5B_Q2_K"]["rho"]),
    ]:
        kl_synth = np.clip(rho_val*b_all + rng.normal(0, 0.3, len(b_all)), 0, None)
        ax.scatter(b_all, kl_synth, s=14, alpha=0.5, color=c_color, label=f"{q} (illustrative)")
        z = np.polyfit(b_all, kl_synth, 1); xx = np.linspace(b_all.min(), b_all.max(), 50)
        ax.plot(xx, np.polyval(z, xx), color=c_color, lw=1.5)
ax.set_xlabel("IRT item difficulty b"); ax.set_ylabel("KL(full || quant) per item")
ax.set_title("C  Probes 2\u20133: KL disturbance vs difficulty\n(within- vs cross-family)")
ax.legend(fontsize=8)

# --- D: IRT vs averaging (summary from measured findings.json or placeholders) ---
ax = axs[1, 1]
dense_rho = F["F6_irt_vs_averaging"]["delta_rho_dense"]; sparse_rho = F["F6_irt_vs_averaging"]["delta_rho_sparse"]
vals = [dense_rho, sparse_rho]
bars = ax.bar(["Dense\n(13x100)", "Sparse\n(simulated)"], vals,
              color=["#4c72b0", "#dd8452"], width=0.5, edgecolor="black", lw=0.5)
ymin, ymax = min(vals + [0.0]), max(vals + [0.0])
pad = max(0.015, (ymax - ymin) * 0.25)
ax.set_ylim(ymin - pad, ymax + pad)
for bar, v in zip(bars, vals):
    offset = pad * 0.18
    y = v + offset if v >= 0 else v - offset
    va = "bottom" if v >= 0 else "top"
    ax.text(bar.get_x()+bar.get_width()/2, y, f"{v:+.2f}", ha="center", va=va, fontsize=8, clip_on=True)
ax.set_ylabel("\u0394\u03c1 (IRT \u2212 simple average)")
ax.set_title("D  Probe 4: IRT vs averaging\n(dense vs sparse)")
ax.axhline(0, ls="-", c="gray", lw=0.5)

fig.savefig(os.path.join(root_dir, "figures", "fig1_main.png"), bbox_inches="tight")
print("fig1_main.png saved")

# ============================ FIGURE 2 ======================================
fig2, ax2 = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
if USE_RAW:
    fig2.suptitle("MEASURED PROBE OUTPUTS — Probe 5 from supplied response matrices",
                  fontsize=10, fontweight="bold")
else:
    fig2.suptitle("ILLUSTRATIVE SCHEMATIC — not measured data (Probe 5 layout)",
                  fontsize=10, fontweight="bold", color="#a00000")

if USE_RAW:
    from scipy.stats import spearmanr
    D = raw["D"]; ids = raw["ids"]; avail = raw["avail"]
    # independent difficulty from ext_3B (negative correct-logprob = harder)
    ext_path = os.path.join(results_dir, "ext_3B.jsonl")
    if os.path.exists(ext_path):
        E = _load("ext_3B")
        gt = -np.array([E[i]["lp_correct"] for i in ids])
        gt_label = "independent difficulty (3B, high=hard)"
    else:
        raise RuntimeError("Raw Figure 2 requires results/ext_3B.jsonl (the independent "
                           "validator). Refusing to mislabel panel difficulty as 3B difficulty.")
    low_subs = [s for s in avail if "Q2" in s or "Q3" in s]
    high_subs = [s for s in avail if "Q8" in s or s == "f16"]
    Xlow = np.array([[D[s][i]["correct"] for i in ids] for s in low_subs], float)
    Xhigh = np.array([[D[s][i]["correct"] for i in ids] for s in high_subs], float)
    dB = _fit_2pl(Xlow) - _fit_2pl(Xhigh)
    rr = spearmanr(gt, dB).correlation
    # 5-fold CV
    rng2 = np.random.RandomState(1)
    folds = np.array_split(rng2.permutation(len(ids)), 5)
    fr = [spearmanr(gt[f], dB[f]).correlation for f in folds]
    ci = F["F5_irt_dif"]["ci"]
else:
    rng = np.random.RandomState(42)
    n = 100
    gt = -np.sort(rng.normal(0, 1, n))[::-1]
    dB = F["F5_irt_dif"]["spearman_dB_vs_3Bdiff"]*gt + rng.normal(0, 0.6, n)
    rr = F["F5_irt_dif"]["spearman_dB_vs_3Bdiff"]
    fr = F["F5_irt_dif"]["cv_folds"]
    ci = F["F5_irt_dif"]["ci"]
    gt_label = "independent difficulty (3B, high=hard)"

ax2[0].scatter(gt, dB, s=16, alpha=0.55, color="#8c564b")
z = np.polyfit(gt, dB, 1); xx = np.linspace(gt.min(), gt.max(), 50)
ax2[0].plot(xx, np.polyval(z, xx), color="#8c564b", lw=1.4)
ax2[0].axhline(0, ls=":", c="gray", lw=0.8)
ax2[0].set_xlabel(gt_label)
ax2[0].set_ylabel("\u0394b = b(low-prec) \u2212 b(high-prec)")
ax2[0].set_title("A  Probe 5: DIF shift \u0394b vs difficulty\n(decision stat: slope + CI)")

ax2[1].bar(range(1, len(fr)+1), fr, color=["#2ca02c" if v > 0 else "#d62728" for v in fr])
ax2[1].axhline(0, c="k", lw=0.8)
ax2[1].set_xlabel("CV fold"); ax2[1].set_ylabel("Spearman(\u0394b, 3B-diff)")
ax2[1].set_title("B  Probe 5: CV fold stability")
lo, hi = min(fr + [0.0]), max(fr + [0.0])
pad = max(0.08, (hi - lo) * 0.25)
ax2[1].set_ylim(max(-1.0, lo - pad), min(1.0, hi + pad))

fig2.savefig(os.path.join(root_dir, "figures", "fig2_dif.png"), bbox_inches="tight")
print("fig2_dif.png saved")
