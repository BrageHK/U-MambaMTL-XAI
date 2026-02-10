from shared_modules.data_module import DataModule
from shared_modules.utils import load_config

from trainer import LitModel
import torch
import json
from tqdm import tqdm
from captum.attr import Occlusion
from captum.attr import Saliency
from pathlib import Path
import time

OUTPUT_DIR = Path("xai_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Settings:
dataset="picai"
label_key = "pca"
config = load_config("config.yaml")
config.gpus = [0]
gpu = config.gpus[0]
config.cache_rate = 1.0
config.transforms.label_keys = ["pca", "prostate_pred", "zones"]
checkpoint_path = "/cluster/home/bragehk/U-MambaMTL-XAI/gc_algorithms/base_container/models/umamba_mtl/weights/f0.ckpt"
model = LitModel.load_from_checkpoint(checkpoint_path, config=config)

model = model.eval()
model.to(0)

def agg_segmentation_wrapper(inp):
    model_out = model(inp)
    out_max = model_out.argmax(dim=1, keepdim=True)
    selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
    aggregated_logits = (model_out * selected_inds).sum(dim=(2, 3, 4))
    return aggregated_logits

occlusion = Occlusion(agg_segmentation_wrapper)
attribute_fn = Saliency(agg_segmentation_wrapper)

size = 20
sliding_window_shapes = (1, size, size, 1)
strides = (1, size, size, 1)
baselines = 0
perturbations_per_eval = 1

dm = DataModule(
    config=config
)
dm.setup("validation")
dl = dm.val_dataloader()

# Collect all per-sample results here
all_results = []

for sample_idx, batch in enumerate(tqdm(dl)):
    is_pca = batch["pca"].max() > 0

    # Extract case ID from MONAI metadata
    case_id = batch["image"].meta["filename_or_obj"][0].split("/")[-1]

    x = batch["image"].to(gpu)
    logits = model(x)

    # ----- Calculate confidence -----
    confidence = round(torch.sigmoid(logits)[0, 1].max().item() * 100, 2)

    # ----- Determine classification -----
    predicted_positive = (torch.sigmoid(logits[:, 1]) > 0.5).any().item()

    if predicted_positive and is_pca:
        classification = "tp"
    elif predicted_positive and not is_pca:
        classification = "fp"
    elif not predicted_positive and not is_pca:
        classification = "tn"
    else:
        classification = "fn"

    # ----- Which zone is pca in? -----
    pca_in_pz = int((batch["pca"] * batch["zones"] == 1).sum().item())
    pca_in_tz = int((batch["pca"] * batch["zones"] == 2).sum().item())

    # ----- Build result record -----
    result = {
        "sample_idx": sample_idx,
        "case_id": case_id,
        "classification": classification,
        "has_pca": bool(is_pca),
        "predicted_positive": bool(predicted_positive),
        "confidence": confidence,
        "pca_voxels_in_pz": pca_in_pz,
        "pca_voxels_in_tz": pca_in_tz,
    }

    # ----- Calculate & save XAI attributions (only for positive predictions) -----
    if predicted_positive:
        sample_dir = OUTPUT_DIR / f"sample_{sample_idx:04d}_{case_id}"
        sample_dir.mkdir(exist_ok=True)

        saliency_map = attribute_fn.attribute(x, target=1, abs=True)

        t0 = time.time()
        occlusion_map = occlusion.attribute(
            x,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            target=1,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=True
        )
        print(f"Occlusion took {time.time() - t0:.2f}s")

        # Save maps as compressed tensors
        torch.save(saliency_map.cpu(), sample_dir / "saliency_map.pt")
        torch.save(occlusion_map.cpu(), sample_dir / "occlusion_map.pt")

        result["maps_dir"] = str(sample_dir)
    else:
        result["maps_dir"] = None

    print(f"[{sample_idx}] {case_id}: {classification.upper()} | confidence={confidence}% | pz={pca_in_pz} tz={pca_in_tz}")
    all_results.append(result)

    # Save incrementally so we don't lose progress on crash
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

print(f"\nDone. Saved {len(all_results)} results to {OUTPUT_DIR / 'results.json'}")
print(f"  TP: {sum(1 for r in all_results if r['classification'] == 'tp')}")
print(f"  FP: {sum(1 for r in all_results if r['classification'] == 'fp')}")
print(f"  TN: {sum(1 for r in all_results if r['classification'] == 'tn')}")
print(f"  FN: {sum(1 for r in all_results if r['classification'] == 'fn')}")
