"""Task 1b — Data exploration.

Reads the four partition CSVs and reports:
  - Image counts per partition
  - Case overlap across partitions (cross-plane pairing potential)
  - Available files on disk
  - Volume shapes (to verify U-Net padding requirements)
Saves results to task_1b/results/01_exploration.json.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    CSV_NONOISE_NOMOTION, CSV_NONOISE_WITHMOTION,
    CSV_WITHNOISE_NOMOTION, CSV_WITHNOISE_WITHMOTION,
    TRAIN_DIR, RESULTS_DIR,
)
from task_1b.utils.io import find_image_path, load_volume

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_case(fn: str) -> str:
    m = re.search(r'(?i)lisa_(\d+)_', fn)
    return m.group(1) if m else '?'


def load_partition(csv_path: Path, label: str):
    if not csv_path.exists():
        print(f"  [!] CSV not found: {csv_path}")
        return pd.DataFrame(columns=['filename'])
    df = pd.read_csv(csv_path)
    df['partition'] = label
    df['case_id']   = df['filename'].apply(extract_case)
    print(f"  {label:30s}: {len(df):4d} images")
    return df


def main():
    print("=" * 60)
    print("Task 1b — Data Exploration")
    print("=" * 60)

    partitions = {
        'nonoise_nomotion':   CSV_NONOISE_NOMOTION,
        'nonoise_withmotion': CSV_NONOISE_WITHMOTION,
        'withnoise_nomotion': CSV_WITHNOISE_NOMOTION,
        'withnoise_withmotion': CSV_WITHNOISE_WITHMOTION,
    }

    dfs = {}
    print("\n── Partition sizes ──────────────────────────────────")
    for label, csv_path in partitions.items():
        dfs[label] = load_partition(csv_path, label)

    all_df = pd.concat(dfs.values(), ignore_index=True) if any(len(d) > 0 for d in dfs.values()) else pd.DataFrame()
    print(f"  {'TOTAL':30s}: {len(all_df):4d} images")

    # Cross-partition case overlap (pairing potential)
    print("\n── Cross-partition case overlap ──────────────────────")
    case_parts = defaultdict(set)
    for label, df in dfs.items():
        for cid in df['case_id']:
            case_parts[cid].add(label)

    has_clean = {cid for cid, parts in case_parts.items() if 'nonoise_nomotion' in parts}
    has_noisy = {cid for cid, parts in case_parts.items()
                 if any(p != 'nonoise_nomotion' for p in parts)}
    paired_cases = has_clean & has_noisy

    print(f"  Cases with nonoise_nomotion images  : {len(has_clean)}")
    print(f"  Cases with noisy/motion images      : {len(has_noisy)}")
    print(f"  Cases with BOTH (pairing potential) : {len(paired_cases)}")

    # File availability check
    print("\n── File availability on disk ─────────────────────────")
    avail, missing, shapes = {}, {}, {}
    for label, df in dfs.items():
        found, lost = 0, 0
        for fn in df['filename']:
            p = find_image_path(fn, TRAIN_DIR)
            if p:
                found += 1
                if found <= 3:  # sample a few shapes
                    try:
                        vol, _, thin_axis = load_volume(p)
                        shapes[fn] = {
                            'shape': list(vol.shape),
                            'thin_axis': int(thin_axis),
                            'n_slices': int(vol.shape[thin_axis]),
                        }
                    except Exception as e:
                        shapes[fn] = {'error': str(e)}
            else:
                lost += 1
        avail[label]  = found
        missing[label] = lost
        print(f"  {label:30s}: {found} found, {lost} missing")

    # Slice counts (for dataset size estimate)
    print("\n── Estimated slice count from available nonoise_nomotion ──")
    nn_df = dfs.get('nonoise_nomotion', pd.DataFrame())
    total_slices = 0
    sampled = 0
    for fn in nn_df['filename']:
        p = find_image_path(fn, TRAIN_DIR)
        if p and sampled < 20:
            try:
                vol, _, thin_axis = load_volume(p)
                n = vol.shape[thin_axis]
                total_slices += n
                sampled += 1
            except Exception:
                pass
    if sampled > 0:
        avg_slices = total_slices / sampled
        estimated = int(avg_slices * avail.get('nonoise_nomotion', 0))
        print(f"  Avg slices/volume (sample n={sampled}): {avg_slices:.1f}")
        print(f"  Estimated training slices (nonoise_nomotion): ~{estimated}")
    else:
        avg_slices = 0
        estimated  = 0
        print("  (no files available yet — cannot estimate)")

    # Sample shapes
    if shapes:
        print("\n── Sample volume shapes ──────────────────────────────")
        for fn, info in list(shapes.items())[:5]:
            print(f"  {fn}: {info}")

    results = {
        'partition_sizes': {k: len(v) for k, v in dfs.items()},
        'files_available': avail,
        'files_missing':   missing,
        'paired_cases':    len(paired_cases),
        'avg_slices_per_volume': float(avg_slices) if avg_slices else None,
        'estimated_train_slices': estimated,
        'sample_shapes': list(shapes.items())[:10],
    }
    out = RESULTS_DIR / '01_exploration.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
