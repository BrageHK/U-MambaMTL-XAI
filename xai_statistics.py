import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser(description="XAI statistics across all folds for a given model")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g. umamba_mtl, swin_unetr)")
args = parser.parse_args()

model = args.model
MODEL_DIR = Path(f"xai_outputs/{model}")

# Discover all fold directories
fold_dirs = sorted(MODEL_DIR.glob("f[0-9]*"))
if not fold_dirs:
    raise FileNotFoundError(f"No fold directories found in {MODEL_DIR}")

print(f"Model: {model}")
print(f"Found {len(fold_dirs)} fold(s): {[d.name for d in fold_dirs]}")

# Concatenate results from all folds
results = []
for fold_dir in fold_dirs:
    results_path = fold_dir / "results.json"
    if not results_path.exists():
        print(f"  WARNING: {results_path} not found, skipping")
        continue
    with open(results_path) as f:
        fold_results = json.load(f)
    print(f"  {fold_dir.name}: {len(fold_results)} samples")
    results.extend(fold_results)

print(f"Total samples across all folds: {len(results)}\n")

OUTPUT_DIR = MODEL_DIR
CHANNEL_NAMES = ["T2W", "ADC", "DWI"]

# Extract ground truth and predictions
y_true = [r["has_pca"] for r in results]
y_pred = [r["predicted_positive"] for r in results]

# Build confusion matrix
labels = [False, True]
cm = confusion_matrix(y_true, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(f"{model} — PCa Detection Confusion Matrix (all folds)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)

tn, fp, fn, tp = cm.ravel()
print(f"TP: {tp}  FP: {fp}")
print(f"FN: {fn}  TN: {tn}")
print(f"Accuracy:    {(tp + tn) / (tp + tn + fp + fn):.4f}")
print(f"Sensitivity: {tp / (tp + fn):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")
print(f"Precision:   {tp / (tp + fp):.4f}")
print(f"F1 Score:    {2 * tp / (2 * tp + fp + fn):.4f}")
print(f"Total samples: {len(results)}")

# ---- Tumor zone distribution (PZ vs TZ) ----
pca_samples = [r for r in results if r["has_pca"]]
for sample in pca_samples:
    print(sample["pca_voxels_in_pz"], sample["pca_voxels_in_tz"])
pz_only = [r for r in pca_samples if r["pca_voxels_in_pz"] > 0 and r["pca_voxels_in_tz"] == 0]
tz_only = [r for r in pca_samples if r["pca_voxels_in_tz"] > 0 and r["pca_voxels_in_pz"] == 0]
both = [r for r in pca_samples if r["pca_voxels_in_pz"] > 0 and r["pca_voxels_in_tz"] > 0]

print(f"\n--- Tumor Zone Distribution (n={len(pca_samples)} PCa cases) ---")
print(f"PZ only:  {len(pz_only):3d} ({len(pz_only)/len(pca_samples)*100:.1f}%)")
print(f"TZ only:  {len(tz_only):3d} ({len(tz_only)/len(pca_samples)*100:.1f}%)")
print(f"Both:     {len(both):3d} ({len(both)/len(pca_samples)*100:.1f}%)")

# Detection rate per zone
tp_pz = [r for r in pz_only if r["classification"] == "tp"]
tp_tz = [r for r in tz_only if r["classification"] == "tp"]
tp_both = [r for r in both if r["classification"] == "tp"]

print(f"\n--- Detection Rate by Zone ---")
if pz_only:
    print(f"PZ only:  {len(tp_pz)}/{len(pz_only)} detected ({len(tp_pz)/len(pz_only)*100:.1f}%)")
if tz_only:
    print(f"TZ only:  {len(tp_tz)}/{len(tz_only)} detected ({len(tp_tz)/len(tz_only)*100:.1f}%)")
if both:
    print(f"Both:     {len(tp_both)}/{len(both)} detected ({len(tp_both)/len(both)*100:.1f}%)")

# Bar chart
zone_labels = ["PZ only", "TZ only", "Both zones"]
zone_counts = [len(pz_only), len(tz_only), len(both)]
zone_detected = [len(tp_pz), len(tp_tz), len(tp_both)]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(zone_labels))
w = 0.35
ax.bar(x - w/2, zone_counts, w, label="Total", color="#6baed6")
ax.bar(x + w/2, zone_detected, w, label="Detected (TP)", color="#2171b5")
ax.set_xticks(x)
ax.set_xticklabels(zone_labels)
ax.set_ylabel("Number of cases")
ax.set_title("Tumor Zone Distribution & Detection Rate")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "zone_distribution.png", dpi=150)

# ============================================================
# Channel-wise XAI Statistics (Saliency & Occlusion maps)
# ============================================================
samples_with_maps = [r for r in results if r["maps_dir"] is not None]
print(f"\n{'='*60}")
print(f"Channel-wise XAI Statistics ({len(samples_with_maps)} samples with maps)")
print(f"{'='*60}")

# Accumulators per map type
per_sample_records = []

for r in samples_with_maps:
    sample_dir = Path(r["maps_dir"])
    saliency_path = sample_dir / "saliency_map.pt"
    occlusion_path = sample_dir / "occlusion_map.pt"

    if not saliency_path.exists() or not occlusion_path.exists():
        print(saliency_path)
        print(occlusion_path)
        print("Whoops skipping")
        continue

    sal = torch.load(saliency_path, weights_only=False)[0]  # [3, H, W, D]
    occ = torch.load(occlusion_path, weights_only=False)[0]  # [3, H, W, D]

    record = {
        "case_id": r["case_id"],
        "classification": r["classification"],
    }

    for map_name, m in [("saliency", sal), ("occlusion", occ)]:
        ch_mean = m.mean(dim=(1, 2, 3)).numpy()  # [3]
        ch_max = m.amax(dim=(1, 2, 3)).numpy()    # [3]
        ch_sum = m.sum(dim=(1, 2, 3)).numpy()     # [3]
        ch_std = m.std(dim=(1, 2, 3)).numpy()     # [3]

        total_activation = ch_sum.sum()
        ch_fraction = ch_sum / total_activation if total_activation > 0 else ch_sum * 0

        dominant_ch = int(np.argmax(ch_mean))

        record[f"{map_name}_ch_mean"] = ch_mean
        record[f"{map_name}_ch_max"] = ch_max
        record[f"{map_name}_ch_sum"] = ch_sum
        record[f"{map_name}_ch_std"] = ch_std
        record[f"{map_name}_ch_fraction"] = ch_fraction
        record[f"{map_name}_dominant_ch"] = dominant_ch

    per_sample_records.append(record)

print(f"Loaded maps for {len(per_sample_records)} samples.\n")

print(per_sample_records)

# --- Aggregate statistics ---
for map_name in ["saliency", "occlusion"]:
    all_means = np.stack([r[f"{map_name}_ch_mean"] for r in per_sample_records])
    all_maxs = np.stack([r[f"{map_name}_ch_max"] for r in per_sample_records])
    all_stds = np.stack([r[f"{map_name}_ch_std"] for r in per_sample_records])
    all_fracs = np.stack([r[f"{map_name}_ch_fraction"] for r in per_sample_records])
    dominant_channels = [r[f"{map_name}_dominant_ch"] for r in per_sample_records]

    print(f"--- {map_name.upper()} Map Channel Statistics ---")
    print(f"{'Channel':<8} {'Avg Activation':>16} {'Avg Max':>12} {'Avg Std':>12} {'Avg Fraction':>14}")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"{name:<8} {all_means[:, i].mean():>16.6f} {all_maxs[:, i].mean():>12.4f} "
              f"{all_stds[:, i].mean():>12.6f} {all_fracs[:, i].mean():>13.1%}")

    # Most important channel (by mean activation) across all samples
    dominant_counts = [dominant_channels.count(i) for i in range(3)]
    print(f"\nDominant channel (highest mean activation per sample):")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name}: {dominant_counts[i]:3d} / {len(per_sample_records)} "
              f"({dominant_counts[i]/len(per_sample_records)*100:.1f}%)")
    print()

    # Breakdown by classification (TP vs FP)
    for cls in ["tp", "fp"]:
        cls_records = [r for r in per_sample_records if r["classification"] == cls]
        if not cls_records:
            continue
        cls_means = np.stack([r[f"{map_name}_ch_mean"] for r in cls_records])
        cls_fracs = np.stack([r[f"{map_name}_ch_fraction"] for r in cls_records])
        cls_dominant = [r[f"{map_name}_dominant_ch"] for r in cls_records]

        print(f"  {cls.upper()} samples (n={len(cls_records)}):")
        print(f"  {'Channel':<8} {'Avg Activation':>16} {'Avg Fraction':>14}")
        for i, name in enumerate(CHANNEL_NAMES):
            print(f"  {name:<8} {cls_means[:, i].mean():>16.6f} {cls_fracs[:, i].mean():>13.1%}")
        dom_counts = [cls_dominant.count(i) for i in range(3)]
        print(f"  Dominant: {', '.join(f'{CHANNEL_NAMES[i]}={dom_counts[i]}' for i in range(3))}")
        print()

# ============================================================
# Plots
# ============================================================

# 1) Average activation per channel (saliency vs occlusion side by side)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, map_name in zip(axes, ["saliency", "occlusion"]):
    all_means = np.stack([r[f"{map_name}_ch_mean"] for r in per_sample_records])
    avg = all_means.mean(axis=0)
    std = all_means.std(axis=0)

    bars = ax.bar(CHANNEL_NAMES, avg, yerr=std, capsize=5, color=["#e74c3c", "#2ecc71", "#3498db"])
    ax.set_ylabel("Mean Activation")
    ax.set_title(f"{map_name.capitalize()} — Avg Activation per Channel")
    ax.bar_label(bars, fmt="%.4f", padding=3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "channel_avg_activation.png", dpi=150)
print("Saved channel_avg_activation.png")

# 2) Dominant channel pie chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, map_name in zip(axes, ["saliency", "occlusion"]):
    dominant = [r[f"{map_name}_dominant_ch"] for r in per_sample_records]
    counts = [dominant.count(i) for i in range(3)]
    ax.pie(counts, labels=CHANNEL_NAMES, autopct="%1.1f%%",
           colors=["#e74c3c", "#2ecc71", "#3498db"], startangle=90)
    ax.set_title(f"{map_name.capitalize()} — Dominant Channel")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "channel_dominance.png", dpi=150)
print("Saved channel_dominance.png")

# 3) Channel fraction distribution (box plot)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, map_name in zip(axes, ["saliency", "occlusion"]):
    all_fracs = np.stack([r[f"{map_name}_ch_fraction"] for r in per_sample_records])
    bp = ax.boxplot([all_fracs[:, i] for i in range(3)], labels=CHANNEL_NAMES, patch_artist=True)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Fraction of Total Activation")
    ax.set_title(f"{map_name.capitalize()} — Channel Fraction Distribution")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "channel_fraction_boxplot.png", dpi=150)
print("Saved channel_fraction_boxplot.png")

# 4) TP vs FP channel comparison
for map_name in ["saliency", "occlusion"]:
    tp_records = [r for r in per_sample_records if r["classification"] == "tp"]
    fp_records = [r for r in per_sample_records if r["classification"] == "fp"]
    if not tp_records or not fp_records:
        continue

    tp_means = np.stack([r[f"{map_name}_ch_mean"] for r in tp_records]).mean(axis=0)
    fp_means = np.stack([r[f"{map_name}_ch_mean"] for r in fp_records]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, tp_means, w, label="TP", color="#2171b5")
    ax.bar(x + w/2, fp_means, w, label="FP", color="#cb181d")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES)
    ax.set_ylabel("Mean Activation")
    ax.set_title(f"{map_name.capitalize()} — TP vs FP Channel Activation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"channel_tp_vs_fp_{map_name}.png", dpi=150)
    print(f"Saved channel_tp_vs_fp_{map_name}.png")

# Save per-sample channel stats to JSON
channel_stats_export = []
for r in per_sample_records:
    entry = {
        "case_id": r["case_id"],
        "classification": r["classification"],
    }
    for map_name in ["saliency", "occlusion"]:
        for i, ch_name in enumerate(CHANNEL_NAMES):
            entry[f"{map_name}_{ch_name}_mean"] = float(r[f"{map_name}_ch_mean"][i])
            entry[f"{map_name}_{ch_name}_max"] = float(r[f"{map_name}_ch_max"][i])
            entry[f"{map_name}_{ch_name}_std"] = float(r[f"{map_name}_ch_std"][i])
            entry[f"{map_name}_{ch_name}_fraction"] = float(r[f"{map_name}_ch_fraction"][i])
        entry[f"{map_name}_dominant_channel"] = CHANNEL_NAMES[r[f"{map_name}_dominant_ch"]]
    channel_stats_export.append(entry)

with open(OUTPUT_DIR / "channel_statistics.json", "w") as f:
    json.dump(channel_stats_export, f, indent=2)
print(f"\nSaved per-sample channel statistics to channel_statistics.json")
