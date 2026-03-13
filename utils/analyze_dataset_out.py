import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
DATASET_OUT = r"D:\USA_paper\dataset\emit_cuprite_dataset_out"

SPLITS = ["train", "val", "test"]

MAX_SAMPLES_FOR_BAND_STATS = 8000
MAX_PER_CLASS_FOR_PLOTS    = 1500

# --- Gap detection (segment-based) ---
# In your plots, the "gap" plateau max is ~0 or slightly negative.
# We'll detect gap bands by max_band < GAP_MAX_THR and (optionally) low std.
GAP_MAX_THR   = 0.10      # if max value in that band is below this -> likely gap
GAP_STD_THR   = 0.01      # if std is below this -> supports gap (optional)
USE_STD_IN_GAP = True     # recommended True
DILATE_BANDS  = 2         # expand each gap segment by ± this many bands to remove vertical walls

# =============================
# IO Helpers
# =============================
def load_meta():
    with open(os.path.join(DATASET_OUT, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def iter_split_npz_paths(split):
    split_file = os.path.join(DATASET_OUT, "splits", f"{split}.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel:
                yield os.path.join(DATASET_OUT, rel), rel

def load_split_labels(split):
    ys = []
    for abs_path, _ in iter_split_npz_paths(split):
        d = np.load(abs_path)
        ys.append(int(d["y"]))
    return np.array(ys, dtype=np.int32)

# =============================
# Class distribution
# =============================
def class_distribution():
    for split in SPLITS:
        ys = load_split_labels(split)
        uniq, cnt = np.unique(ys, return_counts=True)
        total = cnt.sum()
        print(f"\n[{split}] class distribution (count / percent):")
        for u, c in zip(uniq, cnt):
            print(f"  class {int(u)}: {int(c)}  ({100.0*c/total:.2f}%)")

# =============================
# Collect center spectra samples
# =============================
def collect_center_spectra(split="train", max_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    all_paths = [p for p in iter_split_npz_paths(split)]
    if len(all_paths) == 0:
        raise RuntimeError(f"No samples found in split={split}")

    n = min(max_samples, len(all_paths))
    idxs = rng.choice(len(all_paths), size=n, replace=False)

    specs = []
    rels = []
    for i in idxs:
        abs_path, rel = all_paths[i]
        d = np.load(abs_path)
        x = d["x"].astype(np.float32)  # (B,P,P)
        P = x.shape[1]
        c = P // 2
        s = x[:, c, c]                 # (B,)
        specs.append(s)
        rels.append(rel)

    S = np.stack(specs, axis=0)  # (N,B)
    return S, rels

# =============================
# Segment-based gap detection
# =============================
def _segments_from_bool(mask):
    """Return list of (start, end) inclusive segments where mask is True."""
    segs = []
    in_seg = False
    s = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            s = i
        elif (not v) and in_seg:
            in_seg = False
            segs.append((s, i - 1))
    if in_seg:
        segs.append((s, len(mask) - 1))
    return segs

def build_gap_mask_by_segments(max_band, std_band=None,
                               gap_max_thr=0.10, gap_std_thr=0.01,
                               use_std=True, dilate=2):
    """
    Detect gap bands as contiguous segments:
      base_mask = (max_band < gap_max_thr) AND (std_band < gap_std_thr) [optional]
    Then dilate each segment by ±dilate bands.
    """
    max_band = np.asarray(max_band)
    base = (max_band < gap_max_thr)

    if use_std and (std_band is not None):
        std_band = np.asarray(std_band)
        base = base & (std_band < gap_std_thr)

    segs = _segments_from_bool(base)
    gap = np.zeros_like(base, dtype=bool)

    B = len(base)
    for (a, b) in segs:
        aa = max(0, a - dilate)
        bb = min(B - 1, b + dilate)
        gap[aa:bb + 1] = True

    return gap, segs

# =============================
# Interpolation (blockwise)
# =============================
def fill_gaps_blockwise(y, gap_mask):
    y = y.astype(np.float32).copy()
    x = np.arange(len(y))

    valid = (~gap_mask) & np.isfinite(y)
    if valid.sum() < 2:
        return y

    y[gap_mask] = np.interp(x[gap_mask], x[valid], y[valid]).astype(np.float32)
    return y

# =============================
# Plot helpers
# =============================
def plot_per_band_minmax(min_band, max_band, split="train"):
    B = len(min_band)
    x = np.arange(B)
    plt.figure(figsize=(12, 4))
    plt.plot(x, min_band, label="min")
    plt.plot(x, max_band, label="max")
    plt.title(f"Per-band min/max (sampled center spectra) - split={split}")
    plt.xlabel("Band")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_per_band_std(std_band, split="train"):
    B = len(std_band)
    x = np.arange(B)
    plt.figure(figsize=(12, 4))
    plt.plot(x, std_band, label="std")
    plt.title(f"Per-band std (sampled center spectra) - split={split}")
    plt.xlabel("Band")
    plt.ylabel("Std")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_gap_mask(gap_mask, segs):
    plt.figure(figsize=(12, 1.8))
    plt.imshow(gap_mask[None, :], aspect="auto")
    plt.yticks([])
    plt.title(f"Detected gap bands (True=gap) | segments={segs}")
    plt.tight_layout()
    plt.show()

def plot_random_single_spectra(S, rels, k=3, split="train"):
    rng = np.random.default_rng(0)
    idxs = rng.choice(S.shape[0], size=min(k, S.shape[0]), replace=False)

    plt.figure(figsize=(12, 4))
    for j, idx in enumerate(idxs):
        plt.plot(S[idx], label=f"sample {j+1}: {rels[idx]}")
    plt.title(f"Random single spectra (center pixel) - split={split}")
    plt.xlabel("Band")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# =============================
# Class spectra aggregation + plots
# =============================
def collect_class_spectra(split="train", max_per_class=1500):
    spectra_by_class = {}
    counts = {}

    for abs_path, _rel in iter_split_npz_paths(split):
        d = np.load(abs_path)
        x = d["x"].astype(np.float32)  # (B,P,P)
        y = int(d["y"])

        counts[y] = counts.get(y, 0)
        if counts[y] >= max_per_class:
            continue

        P = x.shape[1]
        c = P // 2
        s = x[:, c, c]  # (B,)
        spectra_by_class.setdefault(y, []).append(s)
        counts[y] += 1

    out = {k: np.stack(v, axis=0) for k, v in spectra_by_class.items()}
    return out

def plot_class_spectra(spectra_by_class, gap_mask, with_offset=True, fill_gaps=False, title_prefix=""):
    classes = sorted(spectra_by_class.keys())
    B = spectra_by_class[classes[0]].shape[1]
    x = np.arange(B)

    plt.figure(figsize=(12, 5))
    offset = 0.0

    for k in classes:
        S = spectra_by_class[k]
        mean = np.nanmean(S, axis=0)
        std  = np.nanstd(S, axis=0)

        if fill_gaps:
            mean_plot = fill_gaps_blockwise(mean, gap_mask)
            std_plot  = fill_gaps_blockwise(std,  gap_mask)
        else:
            mean_plot = mean.copy()
            std_plot  = std.copy()
            mean_plot[gap_mask] = np.nan
            std_plot[gap_mask]  = np.nan

        mean_plot = np.where(np.isfinite(mean_plot), mean_plot, np.nan)
        std_plot  = np.where(np.isfinite(std_plot),  std_plot,  np.nan)

        if with_offset:
            plt.plot(x, mean_plot + offset, label=f"class {k} (n={S.shape[0]})")
            plt.fill_between(x, mean_plot - std_plot + offset, mean_plot + std_plot + offset, alpha=0.2)
            offset += 0.2
        else:
            plt.plot(x, mean_plot, label=f"class {k} (n={S.shape[0]})")
            plt.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.2)

    base = "Average Spectral Signatures with Std Dev" if with_offset else "Average Spectral Signatures (No Offset)"
    if title_prefix:
        base = f"{base} | {title_prefix}"

    plt.title(base)
    plt.xlabel("Band")
    plt.ylabel("Value (center pixel)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    meta = load_meta()
    print("META:", meta)

    # 1) Distribution
    class_distribution()

    # 2) Sample spectra for band stats / gap detection
    S_all, rels = collect_center_spectra(
        split="train",
        max_samples=MAX_SAMPLES_FOR_BAND_STATS,
        seed=42
    )

    min_band = np.nanmin(S_all, axis=0)
    max_band = np.nanmax(S_all, axis=0)
    std_band = np.nanstd(S_all, axis=0)

    plot_per_band_minmax(min_band, max_band, split="train")
    plot_per_band_std(std_band, split="train")
    plot_random_single_spectra(S_all, rels, k=3, split="train")

    # 3) Detect gap segments (THIS is the fix)
    gap_mask, segs = build_gap_mask_by_segments(
        max_band=max_band,
        std_band=std_band,
        gap_max_thr=GAP_MAX_THR,
        gap_std_thr=GAP_STD_THR,
        use_std=USE_STD_IN_GAP,
        dilate=DILATE_BANDS
    )

    print(f"\nDetected gap bands: {gap_mask.sum()} / {len(gap_mask)}")
    print("Gap segments (inclusive band indices):", segs)
    if len(segs) > 0:
        for (a, b) in segs:
            print(f"  seg {a}-{b} | max in seg={np.max(max_band[a:b+1]):.4f} | std in seg={np.mean(std_band[a:b+1]):.6f}")

    plot_gap_mask(gap_mask, segs)

    # DEBUG: show before/after for one example curve
    example_mean = np.nanmean(S_all, axis=0)
    filled_example = fill_gaps_blockwise(example_mean, gap_mask)
    print("\nDEBUG example mean around first gap (if exists):")
    if len(segs) > 0:
        a, b = segs[0]
        aa = max(0, a-3)
        bb = min(len(example_mean)-1, b+3)
        print("bands:", list(range(aa, bb+1)))
        print("before:", np.round(example_mean[aa:bb+1], 4))
        print("after :", np.round(filled_example[aa:bb+1], 4))

    # 4) Per-class plots
    spectra_by_class = collect_class_spectra(split="train", max_per_class=MAX_PER_CLASS_FOR_PLOTS)

    # RAW (gaps shown)
    plot_class_spectra(spectra_by_class, gap_mask, with_offset=True,  fill_gaps=False, title_prefix="RAW (gaps shown) (train)")
    plot_class_spectra(spectra_by_class, gap_mask, with_offset=False, fill_gaps=False, title_prefix="RAW (gaps shown) (train)")

    # FILLED (viz only) - NOW it should fill
    plot_class_spectra(spectra_by_class, gap_mask, with_offset=True,  fill_gaps=True, title_prefix="FILLED (viz only) (train)")
    plot_class_spectra(spectra_by_class, gap_mask, with_offset=False, fill_gaps=True, title_prefix="FILLED (viz only) (train)")
