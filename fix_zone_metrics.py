"""
Fix pca_voxels_in_pz and pca_voxels_in_tz in results.json files.

The original computation was wrong because batch["zones"] is one-hot encoded
(shape [B,3,H,W,D] with binary values), so multiplying pca * zones and checking
== 2 can never be true. This script recomputes the values from the raw NIfTI
files using the original integer zone labels {0=background, 1=pz, 2=tz}.

Uses MONAI transforms to resample pca and zones to a common spacing before
comparing, since they may have different native resolutions.
"""

import json
import shutil
import torch
from pathlib import Path
from monai.data import load_decathlon_datalist
from monai import transforms
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

XAI_ROOT = Path("xai_outputs")
DATALIST_DIR = Path("json_datalists/picai")
NUM_WORKERS = 20

SPACING = [0.5, 0.5, 3.0]


def compute_zone_overlap(pca_path, zones_path):
    """Load pca and zones, resample to common spacing, compute overlap."""
    resample = transforms.Compose([
        transforms.LoadImaged(keys=["pca", "zones"], ensure_channel_first=True),
        transforms.Spacingd(keys=["pca", "zones"], pixdim=SPACING, mode="nearest"),
    ])

    sample = resample({"pca": pca_path, "zones": zones_path})
    pca_data = sample["pca"]
    zones_data = torch.round(sample["zones"]).long()

    # Crop to overlapping region (volumes may have different extents)
    min_shape = [min(p, z) for p, z in zip(pca_data.shape[1:], zones_data.shape[1:])]
    pca_data = pca_data[:, :min_shape[0], :min_shape[1], :min_shape[2]]
    zones_data = zones_data[:, :min_shape[0], :min_shape[1], :min_shape[2]]

    pca_mask = pca_data > 0
    pca_in_pz = int((pca_mask & (zones_data == 1)).sum().item())
    pca_in_tz = int((pca_mask & (zones_data == 2)).sum().item())
    return pca_in_pz, pca_in_tz


def process_one(case_id, pca_path, zones_path):
    """Worker function for a single case."""
    pca_in_pz, pca_in_tz = compute_zone_overlap(pca_path, zones_path)
    return case_id, pca_in_pz, pca_in_tz


def fix_results_file(results_path: Path):
    # Parse fold from path: xai_outputs/{model}/f{fold}/results.json
    match = re.search(r"/f(\d+)/", str(results_path))
    if not match:
        print(f"  Skipping {results_path}: cannot determine fold")
        return
    fold = int(match.group(1))

    datalist_path = DATALIST_DIR / f"fold_{fold}.json"
    if not datalist_path.exists():
        print(f"  Skipping {results_path}: {datalist_path} not found")
        return

    datalist = load_decathlon_datalist(str(datalist_path), True, "validation")

    backup_path = results_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy2(results_path, backup_path)

    with open(results_path) as f:
        results = json.load(f)

    # Build case_id -> datalist entry mapping
    case_to_entry = {}
    for entry in datalist:
        image_path = entry["image"][0] if isinstance(entry["image"], list) else entry["image"]
        case_id = image_path.split("/")[-1]
        case_to_entry[case_id] = entry

    # Submit all cases to process pool
    futures = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for result in results:
            case_id = result["case_id"]
            if case_id not in case_to_entry:
                print(f"  Warning: {case_id} not found in datalist")
                continue
            entry = case_to_entry[case_id]
            future = pool.submit(process_one, case_id, entry["pca"], entry["zones"])
            futures[future] = None

        # Collect results
        computed = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc=str(results_path), leave=False):
            case_id, pca_in_pz, pca_in_tz = future.result()
            computed[case_id] = (pca_in_pz, pca_in_tz)

    # Update results
    fixed = 0
    for result in results:
        case_id = result["case_id"]
        if case_id not in computed:
            continue
        pca_in_pz, pca_in_tz = computed[case_id]

        if result["pca_voxels_in_pz"] != pca_in_pz or result["pca_voxels_in_tz"] != pca_in_tz:
            fixed += 1

        result["pca_voxels_in_pz"] = pca_in_pz
        result["pca_voxels_in_tz"] = pca_in_tz

        # Clean up maps_dir path
        if result.get("maps_dir") and "../" in result["maps_dir"]:
            result["maps_dir"] = re.sub(r"(\.\./)+", "", result["maps_dir"])

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Fixed {fixed}/{len(results)} entries in {results_path}")


if __name__ == "__main__":
    results_files = sorted(XAI_ROOT.rglob("results.json"))
    print(f"Found {len(results_files)} results files to fix\n")

    for results_path in results_files:
        fix_results_file(results_path)

    print("\nDone.")
