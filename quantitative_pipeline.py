
import json
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import entropy, kruskal
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. Locate files
PROJECT_ROOT = Path.cwd()
REPORT_DIR = PROJECT_ROOT / "reports"

scenario_paths = sorted(REPORT_DIR.glob("scenario_*/*report.json"))
summary_path = REPORT_DIR / "summary_report.json"

if not scenario_paths:
    raise FileNotFoundError("No scenario_* report.json files under ./reports/")
if not summary_path.exists():
    raise FileNotFoundError("reports/summary_report.json not found")

# 2. Helpers
FRAME_KEYS = [
    "pragmatic_ethics", "rawlsian_justice", "kantian_deontology", "justice",
    "pragmatism", "moral_relativism", "care_ethics", "utilitarianism",
    "feminism", "autonomy", "virtue_ethics"
]

def base_model(model_id: str) -> str:
    parts = model_id.rsplit("_", 1)
    return parts[0] if parts[-1].isdigit() else model_id

def cosine_sim(v1, v2):
    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0.0
    return 1.0 - cosine(v1, v2)

# 3. Build runs_df
runs = []
for p in scenario_paths:
    rpt = json.loads(p.read_text())
    scenario = p.parent.name.replace("scenario_", "")
    for model_id, prof in rpt["individual_profiles"].items():
        vec = np.zeros(len(FRAME_KEYS))
        for i, key in enumerate(FRAME_KEYS):
            sem = prof["semantic_analysis"].get(key, 0.0)
            if key == "kantian_deontology":
                dic = prof["dictionary_analysis"].get("deontological", {}).get("normalized_score", 0.0)
            elif key == "utilitarianism":
                dic = prof["dictionary_analysis"].get("consequentialism", {}).get("normalized_score", 0.0)
            else:
                dic = prof["dictionary_analysis"].get(key, {}).get("normalized_score", 0.0)
            vec[i] = sem + 0.4 * dic
        if vec.sum() > 0:
            vec = vec / vec.sum()

        rc = prof["reasoning_analysis"]["metrics"]
        saf = prof["safety_analysis"]
        total_safety = sum(d["score"] for d in saf.values()) or 1
        systemic = saf.get("systemic_risk", {}).get("score", 0) / total_safety

        runs.append({
            "scenario": scenario,
            "model_id": model_id,
            "base_model": base_model(model_id),
            "fw_vec": vec,
            "entropy_bits": float(entropy(vec, base=2)),
            "top2_ratio": float(np.sort(vec)[-1] / np.sort(vec)[-2]) if (vec > 0).sum() > 1 else np.nan,
            "steps": rc["num_reasoning_steps"],
            "density": rc["reasoning_density"],
            "moral_terms": rc["moral_terms_in_reasoning"],
            "systemic_recall": systemic
        })

runs_df = pd.DataFrame(runs)

# 4. Aggregate to base-model
out = defaultdict(list)
for mdl, grp in runs_df.groupby("base_model"):
    out["model"].append(mdl)
    out["entropy_bits"].append(grp["entropy_bits"].mean())
    out["top2_ratio"].append(grp["top2_ratio"].mean())
    out["avg_steps"].append(grp["steps"].mean())
    out["avg_density"].append(grp["density"].mean())
    out["avg_moral_terms"].append(grp["moral_terms"].mean())
    out["safety_systemic_recall"].append(grp["systemic_recall"].mean())
    vecs = np.stack(grp["fw_vec"].to_list())
    sims = [cosine_sim(vecs[i], vecs[j]) for i, j in itertools.combinations(range(len(vecs)), 2)]
    out["cosine_stability"].append(float(np.mean(sims)))

out_df = pd.DataFrame(out).round(3)

# Save
csv_path = PROJECT_ROOT / "quantitative_metrics.csv"
out_df.to_csv(csv_path, index=False)

print("Quantitative metrics per base model:")
print(out_df)
print(f"→ saved to {csv_path}\n")

# 5. Statistical significance testing
metrics = ["entropy_bits", "top2_ratio", "steps", "density", "moral_terms", "systemic_recall"]
print("=== Kruskal-Wallis tests ===")
for m in metrics:
    groups = [grp[m].values for _, grp in runs_df.groupby("base_model")]
    stat, p = kruskal(*groups)
    print(f"{m:15s} : H={stat:.3f}, p={p:.3f}")
print()

print("=== Tukey HSD post-hoc ===")
for m in metrics:
    df = runs_df[["base_model", m]].dropna()
    tukey = pairwise_tukeyhsd(endog=df[m], groups=df["base_model"], alpha=0.05)
    print(f"Metric: {m}")
    print(tukey.summary())

# 6a. Empirical selection of k
from sklearn.metrics import silhouette_score

k_range = range(2, 8)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42).fit(pc2)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(pc2, km.labels_))

plt.figure(figsize=(6,4))
plt.plot(k_range, inertias, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method: Inertia vs. k")
plt.savefig("elbow_curve.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(k_range, silhouettes, marker='s', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. k")
plt.savefig("silhouette_scores.png")
plt.close()

best_k = k_range[int(np.argmax(silhouettes))]
print(f"Optimal k = {best_k}")

# 6. Clustering / PCA
feat = np.vstack(runs_df["fw_vec"].to_list())
feat = np.hstack([feat, runs_df[["systemic_recall"]].values])
scaler = StandardScaler().fit(feat)
feat_s = scaler.transform(feat)

pca = PCA(n_components=2).fit(feat_s)
pc2 = pca.transform(feat_s)

kmeans = KMeans(n_clusters=best_k, random_state=42).fit(pc2)
clusters = kmeans.labels_

plt.figure(figsize=(7,6))
for bm in runs_df["base_model"].unique():
    idx = runs_df["base_model"] == bm
    plt.scatter(pc2[idx,0], pc2[idx,1], label=bm)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig("pca_by_base_model.png")
plt.close()

plt.figure(figsize=(7,6))
plt.scatter(pc2[:,0], pc2[:,1], c=clusters)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("pca_kmeans_clusters.png")
plt.close()

# 7. Behavioral drift
drift = runs_df.groupby("base_model")[metrics].std().round(3)
print("Behavioral drift:")
print(drift)

axes = drift.plot.bar(subplots=True, layout=(2,3), figsize=(16,7), legend=False)
for ax in axes.flatten():
    ax.set_ylabel("Std Dev")
plt.savefig("behavioral_drift_metrics.png")
plt.close()
