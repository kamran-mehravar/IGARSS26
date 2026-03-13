import os
import json
import numpy as np
import matplotlib.pyplot as plt

DATASET_OUT = r"D:\USA_paper\dataset\emit_cuprite_dataset_out"

# =============================
# Helpers
# =============================
def iter_split_npz_paths(split):
    split_file = os.path.join(DATASET_OUT, "splits", f"{split}.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel:
                yield os.path.join(DATASET_OUT, rel), rel

def load_meta():
    with open(os.path.join(DATASET_OUT, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_split_labels(split):
    ys = []
    for abs_path, _rel in iter_split_npz_paths(split):
        d = np.load(abs_path)
        ys.append(int(d["y"]))
    return np.array(ys, dtype=np.int32)

def class_distribution():
    for split in ["train", "val", "test"]:
        ys = load_split_labels(split)
        uniq, cnt = np.unique(ys, return_counts=True)
        total = cnt.sum()
        print(f"\n[{split}] class distribution (count / percent):")
        for u, c in zip(uniq, cnt):
            print(f"  class {int(u)}: {int(c)}  ({100.0*c/total:.2f}%)")

def center_spectrum_from_patch(x):
    # x: (B,P,P)
    P = x.shape[1]
    c = P // 2
    return x[:, c, c]  # (B,)

# =============================
# Band stats (global)
# =============================
def compute_band_stats(split="train", max_samples=5000, eps=1e-8):
    specs = []
    for i, (abs_path, _rel) in enumerate(iter_split_npz_paths(split)):
        if i >= max_samples:
            break
        d = np.load(abs_path)
        x = d["x"].astype(np.float32)
        specs.append(center_spectrum_from_patch(x))

    S = np.stack(specs, axis=0)  # (N,B)
    nan_frac = np.mean(~np.isfinite(S), axis=0)
    zero_frac = np.mean(np.isfinite(S) & (np.abs(S) <= eps), axis=0)
    std = np.nanstd(S, axis=0)
    return zero_frac, nan_frac, std

def print_top_band_stats(zero_frac, nan_frac, std, topk=25):
    print("\nTop bands by ZERO fraction:")
    topz = np.argsort(-zero_frac)[:topk]
    for b in topz:
        print(f"band {b:3d} | zero={zero_frac[b]:.3f} | nan={nan_frac[b]:.3f} | std={std[b]:.6f}")

    print("\nTop bands by NAN fraction:")
    topn = np.argsort(-nan_frac)[:topk]
    for b in topn:
        print(f"band {b:3d} | nan={nan_frac[b]:.3f} | zero={zero_frac[b]:.3f} | std={std[b]:.6f}")

    print("\nTop bands by LOW std (most constant):")
    topl = np.argsort(std)[:topk]
    for b in topl:
        print(f"band {b:3d} | std={std[b]:.6f} | zero={zero_frac[b]:.3f} | nan={nan_frac[b]:.3f}")

def plot_band_quality_curves(zero_frac, nan_frac, std):
    B = len(zero_frac)
    x = np.arange(B)

    plt.figure(figsize=(12, 3))
    plt.plot(x, zero_frac)
    plt.title("Zero fraction per band (center pixel, sampled)")
    plt.xlabel("Spectral band")
    plt.ylabel("zero_frac")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(x, nan_frac)
    plt.title("NaN fraction per band (center pixel, sampled)")
    plt.xlabel("Spectral band")
    plt.ylabel("nan_frac")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(x, std)
    plt.title("Std per band (center pixel, sampled)")
    plt.xlabel("Spectral band")
    plt.ylabel("std")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def build_invalid_band_mask(zero_frac, nan_frac, std, zero_thr=0.95, nan_thr=0.95, std_thr=1e-6):
    # For your current data this will likely only flag band0 and band284
    return (zero_frac >= zero_thr) | (nan_frac >= nan_thr) | (std <= std_thr)

# =============================
# Extra diagnostics (the missing parts)
# =============================
def plot_random_single_spectra(split="train", n=3, seed=42):
    """
    Plots n random single spectra (center pixel), to verify whether step-like
    patterns exist per sample (not only in averages).
    """
    rng = np.random.default_rng(seed)
    paths = [(abs_path, rel) for abs_path, rel in iter_split_npz_paths(split)]
    if len(paths) == 0:
        print("No samples found.")
        return

    picks = rng.choice(len(paths), size=min(n, len(paths)), replace=False)

    plt.figure(figsize=(12, 4))
    for idx, pi in enumerate(picks, start=1):
        abs_path, rel = paths[int(pi)]
        d = np.load(abs_path)
        x = d["x"].astype(np.float32)
        s = center_spectrum_from_patch(x)

        plt.plot(s, label=f"sample {idx}: {rel}")

    plt.title(f"Random single spectra (center pixel) - split={split}")
    plt.xlabel("Band")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

def band_min_max(split="train", max_samples=5000):
    """
    Plots per-band min/max across sampled center spectra.
    If piecewise clamping exists, it will appear here.
    """
    specs = []
    for i, (abs_path, _rel) in enumerate(iter_split_npz_paths(split)):
        if i >= max_samples:
            break
        d = np.load(abs_path)
        x = d["x"].astype(np.float32)
        specs.append(center_spectrum_from_patch(x))

    S = np.stack(specs, axis=0)  # (N,B)

    bmin = np.min(S, axis=0)
    bmax = np.max(S, axis=0)

    plt.figure(figsize=(12, 4))
    plt.plot(bmin, label="min")
    plt.plot(bmax, label="max")
    plt.title(f"Per-band min/max (sampled center spectra) - split={split}")
    plt.xlabel("Band")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================
# Spectral signature plots
# =============================
def collect_class_spectra(split="train", max_per_class=1500, normalize=None):
    """
    normalize:
      None        -> raw spectrum
      'minmax'    -> per-spectrum min-max to [0,1]
      'zscore'    -> per-spectrum z-score
    """
    spectra_by_class = {}
    counts = {}

    for abs_path, _rel in iter_split_npz_paths(split):
        d = np.load(abs_path)
        x = d["x"].astype(np.float32)
        y = int(d["y"])

        counts[y] = counts.get(y, 0)
        if counts[y] >= max_per_class:
            continue

        s = center_spectrum_from_patch(x).astype(np.float32)

        # optional per-spectrum normalization for comparability
        if normalize == "minmax":
            mn, mx = float(np.min(s)), float(np.max(s))
            if mx - mn > 1e-8:
                s = (s - mn) / (mx - mn)
        elif normalize == "zscore":
            mu, sd = float(np.mean(s)), float(np.std(s)) + 1e-6
            s = (s - mu) / sd

        spectra_by_class.setdefault(y, []).append(s)
        counts[y] += 1

    out = {}
    for k, arr_list in spectra_by_class.items():
        out[k] = np.stack(arr_list, axis=0)  # (N,B)
    return out

def plot_class_spectra(spectra_by_class, invalid_band_mask=None, with_offset=True, title=""):
    classes = sorted(spectra_by_class.keys())
    B = spectra_by_class[classes[0]].shape[1]
    x = np.arange(B)

    offset = 0.0
    plt.figure(figsize=(12, 5))

    for k in classes:
        S = spectra_by_class[k]
        mean = np.nanmean(S, axis=0)
        std = np.nanstd(S, axis=0)

        mean_plot = mean.copy()
        std_plot = std.copy()

        if invalid_band_mask is not None:
            mean_plot[invalid_band_mask] = np.nan
            std_plot[invalid_band_mask] = np.nan

        mean_plot = np.where(np.isfinite(mean_plot), mean_plot, np.nan)
        std_plot = np.where(np.isfinite(std_plot), std_plot, np.nan)

        if with_offset:
            plt.plot(mean_plot + offset, label=f"class {k} (n={S.shape[0]})")
            plt.fill_between(x, mean_plot - std_plot + offset, mean_plot + std_plot + offset, alpha=0.2)
            offset += 0.2
        else:
            plt.plot(mean_plot, label=f"class {k} (n={S.shape[0]})")
            plt.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.2)

    plt.title(title)
    plt.xlabel("Spectral Band")
    plt.ylabel("Value (center pixel)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

# =============================
# Main
# =============================
if __name__ == "__main__":
    meta = load_meta()
    print("META:", meta)

    # 1) class distribution
    class_distribution()

    # 2) band stats
    zero_frac, nan_frac, std = compute_band_stats(split="train", max_samples=5000, eps=1e-8)
    print_top_band_stats(zero_frac, nan_frac, std, topk=25)
    plot_band_quality_curves(zero_frac, nan_frac, std)

    # 3) invalid mask (your current data flags ~2 bands)
    invalid_mask = build_invalid_band_mask(zero_frac, nan_frac, std, zero_thr=0.95, nan_thr=0.95, std_thr=1e-6)
    print(f"\nInvalid bands: {invalid_mask.sum()} / {len(invalid_mask)} ({100.0*invalid_mask.sum()/len(invalid_mask):.2f}%)")

    # 4) NEW: diagnostics to verify "step-like" is real per spectrum
    plot_random_single_spectra(split="train", n=3, seed=42)
    band_min_max(split="train", max_samples=5000)

    # 5) Class mean/std plots (raw)
    spectra_raw = collect_class_spectra(split="train", max_per_class=1500, normalize=None)
    plot_class_spectra(
        spectra_raw,
        invalid_band_mask=invalid_mask,
        with_offset=True,
        title="Average Spectral Signatures with Std Dev (RAW, masked only invalid bands)"
    )
    plot_class_spectra(
        spectra_raw,
        invalid_band_mask=invalid_mask,
        with_offset=False,
        title="Average Spectral Signatures (RAW, masked only invalid bands)"
    )

    # 6) OPTIONAL: normalized version for paper-style comparability
    spectra_mm = collect_class_spectra(split="train", max_per_class=1500, normalize="minmax")
    plot_class_spectra(
        spectra_mm,
        invalid_band_mask=invalid_mask,
        with_offset=True,
        title="Average Spectral Signatures with Std Dev (Per-spectrum MinMax, masked invalid bands)"
    )
    plot_class_spectra(
        spectra_mm,
        invalid_band_mask=invalid_mask,
        with_offset=False,
        title="Average Spectral Signatures (Per-spectrum MinMax, masked invalid bands)"
    )
