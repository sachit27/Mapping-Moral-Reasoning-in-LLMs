
# Robustness / ablation study of ethical-framework vectors

import json
import itertools
import warnings
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial.distance import cosine
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Optional – install hdbscan if missing
try:
    import hdbscan
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan"])
    import hdbscan

warnings.filterwarnings("ignore")
sns.set_theme(context="notebook", style="whitegrid")

# Constants and Paths
PROJECT_ROOT = Path.cwd()
REPORT_DIR = PROJECT_ROOT / "reports"
scenario_paths = sorted(REPORT_DIR.glob("scenario_*/*report.json"))
summary_path = REPORT_DIR / "summary_report.json"

FRAME_KEYS = [
    "pragmatic_ethics", "rawlsian_justice", "kantian_deontology", "justice",
    "pragmatism", "moral_relativism", "care_ethics", "utilitarianism",
    "feminism", "autonomy", "virtue_ethics"
]

def base_model(mid: str) -> str:
    parts = mid.rsplit("_", 1)
    return parts[0] if parts[-1].isdigit() else mid

def collect_runs(weight: float) -> pd.DataFrame:
    rows = []
    for p in scenario_paths:
        data = json.loads(p.read_text())
        for m_id, prof in data["individual_profiles"].items():
            v = np.zeros(len(FRAME_KEYS))
            for idx, key in enumerate(FRAME_KEYS):
                sem = prof["semantic_analysis"].get(key, 0.0)
                if key == "kantian_deontology":
                    extra = prof["dictionary_analysis"].get("deontological", {}).get("normalized_score", 0.0)
                elif key == "utilitarianism":
                    extra = prof["dictionary_analysis"].get("consequentialism", {}).get("normalized_score", 0.0)
                else:
                    extra = prof["dictionary_analysis"].get(key, {}).get("normalized_score", 0.0)
                v[idx] = sem + weight * extra
            if v.sum() > 0:
                v = v / v.sum()
            rows.append(dict(base_model=base_model(m_id), **{k: v[i] for i, k in enumerate(FRAME_KEYS)}))
    return pd.DataFrame(rows)

# Collect data
df_base = collect_runs(weight=0.4)
df_ablation = collect_runs(weight=0.0)

# PCA + Clustering
def project_cluster(df, method="kmeans"):
    X = StandardScaler().fit_transform(df[FRAME_KEYS])
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    if method == "kmeans":
        labels = KMeans(n_clusters=3, random_state=42).fit_predict(pcs)
    elif method == "hdbscan":
        labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(pcs)
    else:
        raise ValueError("Unknown method.")
    return pcs, labels

pcs_base, lbl_base = project_cluster(df_base, "kmeans")
pcs_abl, lbl_abl = project_cluster(df_ablation, "kmeans")
pcs_hdb_base, lbl_hdb_base = project_cluster(df_base, "hdbscan")
pcs_hdb_abl, lbl_hdb_abl = project_cluster(df_ablation, "hdbscan")

# Adjusted Rand Index with Bootstrapped CI
ari_point = adjusted_rand_score(lbl_base, lbl_abl)
BOOT = 1000
boot_aris = []
rng = np.random.default_rng(42)
for _ in range(BOOT):
    idx = rng.choice(len(lbl_base), len(lbl_base), replace=True)
    boot_aris.append(adjusted_rand_score(lbl_base[idx], lbl_abl[idx]))
ci_low, ci_high = np.percentile(boot_aris, [2.5, 97.5])

print(f"Adjusted Rand Index = {ari_point:.3f}  (95% CI: {ci_low:.3f} – {ci_high:.3f})")

# Silhouette Scores
sil_base = silhouette_score(pcs_base, lbl_base)
sil_abl = silhouette_score(pcs_abl, lbl_abl)
print(f"Silhouette score – baseline: {sil_base:.3f}   ablation: {sil_abl:.3f}")

# PCA comparison plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
for ax, pcs, labels, title in zip(
    axes,
    [pcs_base, pcs_abl],
    [lbl_base, lbl_abl],
    ["Baseline (ω = 0.4)", "Ablation (ω = 0.0)"]
):
    ax.scatter(pcs[:,0], pcs[:,1], c=labels, cmap="Set1", s=90, edgecolor="k")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_baseline_vs_ablation.png")
plt.close()

# ARI bootstrap distribution
plt.figure(figsize=(8,4.5))
plt.hist(boot_aris, bins=30, alpha=0.8)
plt.axvline(ari_point, color="black", lw=2, label=f"ARI = {ari_point:.3f}")
plt.axvline(ci_low, color="red", linestyle="--", label="95% CI")
plt.axvline(ci_high, color="red", linestyle="--")
plt.legend()
plt.savefig("ari_bootstrap_distribution.png")
plt.close()

# HDBSCAN comparison plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
titles = ["HDBSCAN – Baseline", "HDBSCAN – Ablation"]
for ax, pcs, labels, title in zip(axes, [pcs_hdb_base, pcs_hdb_abl], [lbl_hdb_base, lbl_hdb_abl], titles):
    palette = sns.color_palette("husl", len(np.unique(labels)))
    colors = [palette[l] if l != -1 else "lightgrey" for l in labels]
    ax.scatter(pcs[:,0], pcs[:,1], c=colors, s=90, edgecolor="k")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig("hdbscan_baseline_vs_ablation.png")
plt.close()

# HDBSCAN summary
hdb_ari = adjusted_rand_score(lbl_hdb_base, lbl_hdb_abl)
sil_hdb_base = silhouette_score(pcs_hdb_base, lbl_hdb_base) if len(set(lbl_hdb_base)) > 1 else np.nan
sil_hdb_abl = silhouette_score(pcs_hdb_abl, lbl_hdb_abl) if len(set(lbl_hdb_abl)) > 1 else np.nan
print(f"HDBSCAN ARI = {hdb_ari:.3f}")
print(f"HDBSCAN silhouette – baseline: {sil_hdb_base:.3f}, ablation: {sil_hdb_abl:.3f}")
