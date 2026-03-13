import os
import json
import numpy as np
import rasterio

# =============================
# CONFIG
# =============================
CUBE_TIF   = r"D:\USA_paper\dataset\emit cuprite\emit cuprite\resize-continuum.tif"   # (B,H,W)
LABEL_TIF  = r"D:\USA_paper\dataset\Emit Py folder\klabels10_georef.tif"             # (H,W) labels 0..9
OUT_DIR    = r"D:\USA_paper\dataset\emit_cuprite_dataset_out"

PATCH_SIZE = 15          # odd
SEED       = 42

# Split strategy
USE_BLOCK_SPLIT = True
BLOCK_SIZE = 32          # decrease to 32 if rare classes disappear in val/test

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

# Labels: you confirmed unique values are 0..9 and NO nodata
IGNORE_LABELS = set()     # IMPORTANT: empty
LABEL_OFFSET  = 0         # labels already 0..9

# Data cleaning
# Keep this True only if zeros are truly invalid/masked values in your cube.
ZERO_TO_NAN = True

# Rare-class handling
RARE_CLASSES = {2}          # فعلاً فقط کلاس 2
MIN_VAL_PER_RARE  = 20      # حداقل 20 نمونه از کلاس نادر در val
MIN_TEST_PER_RARE = 20      # حداقل 20 نمونه از کلاس نادر در test


# =============================
# Helpers
# =============================
def ensure_dirs(base):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "splits"), exist_ok=True)
    os.makedirs(os.path.join(base, "patches", "train"), exist_ok=True)
    os.makedirs(os.path.join(base, "patches", "val"), exist_ok=True)
    os.makedirs(os.path.join(base, "patches", "test"), exist_ok=True)

def read_cube(path_tif):
    with rasterio.open(path_tif) as src:
        cube = src.read().astype(np.float32)  # (B,H,W)
        profile = src.profile
    if ZERO_TO_NAN:
        cube[cube == 0] = np.nan
    return cube, profile

def read_labels(path_tif):
    with rasterio.open(path_tif) as src:
        lab = src.read(1).astype(np.int32)
        profile = src.profile

    if LABEL_OFFSET != 0:
        mask = ~np.isin(lab, list(IGNORE_LABELS))
        lab[mask] = lab[mask] - LABEL_OFFSET

    return lab, profile

def infer_classes(labels):
    uniq = sorted(np.unique(labels).tolist())
    uniq = [u for u in uniq if (u not in IGNORE_LABELS) and (u >= 0)]
    n_classes = int(max(uniq) + 1) if uniq else 0
    return n_classes, uniq

def valid_centers(labels, patch_size):
    H, W = labels.shape
    r = patch_size // 2
    centers = []

    ignore_list = list(IGNORE_LABELS)
    for y in range(r, H - r):
        row = labels[y, r:W - r]
        if ignore_list:
            ok = ~np.isin(row, ignore_list)
        else:
            ok = np.ones_like(row, dtype=bool)
        xs = np.where(ok)[0] + r
        for x in xs:
            centers.append((y, x))

    return centers

def block_split(centers, block_size, seed):
    rng = np.random.default_rng(seed)

    blocks = {}
    for (y, x) in centers:
        bid = (y // block_size, x // block_size)
        blocks.setdefault(bid, []).append((y, x))

    block_ids = list(blocks.keys())
    rng.shuffle(block_ids)

    n_blocks = len(block_ids)
    n_train = int(n_blocks * TRAIN_RATIO)
    n_val   = int(n_blocks * VAL_RATIO)

    train_ids = set(block_ids[:n_train])
    val_ids   = set(block_ids[n_train:n_train + n_val])
    test_ids  = set(block_ids[n_train + n_val:])

    train = [pt for bid in train_ids for pt in blocks[bid]]
    val   = [pt for bid in val_ids   for pt in blocks[bid]]
    test  = [pt for bid in test_ids  for pt in blocks[bid]]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test

def random_split(centers, seed):
    rng = np.random.default_rng(seed)
    centers = centers.copy()
    rng.shuffle(centers)

    n = len(centers)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = centers[:n_train]
    val   = centers[n_train:n_train + n_val]
    test  = centers[n_train + n_val:]
    return train, val, test

def save_patches(cube, labels, centers, patch_size, split_name, start_idx, out_dir):
    B, H, W = cube.shape
    r = patch_size // 2

    split_txt = os.path.join(out_dir, "splits", f"{split_name}.txt")
    with open(split_txt, "w", encoding="utf-8") as f:
        idx = start_idx
        for (y, x) in centers:
            y_cls = int(labels[y, x])
            if y_cls in IGNORE_LABELS:
                continue

            patch = cube[:, y - r:y + r + 1, x - r:x + r + 1]  # (B,P,P)

            name = f"sample_{idx:06d}.npz"
            rel_path = os.path.join("patches", split_name, name)
            abs_path = os.path.join(out_dir, rel_path)

            np.savez_compressed(
                abs_path,
                x=patch.astype(np.float32),
                y=np.int32(y_cls),
                row=np.int32(y),
                col=np.int32(x),
            )
            f.write(rel_path + "\n")
            idx += 1

    return idx

def split_class_counts(labels, centers, n_classes):
    cnt = np.zeros((n_classes,), dtype=np.int64)
    for (y, x) in centers:
        c = int(labels[y, x])
        if 0 <= c < n_classes:
            cnt[c] += 1
    return cnt

def print_counts(name, counts):
    total = int(counts.sum())
    print(f"\n[{name}] total={total}")
    for i, c in enumerate(counts.tolist()):
        pct = 100.0 * c / total if total > 0 else 0.0
        print(f"  class {i}: {c} ({pct:.2f}%)")

def centers_by_class(labels, centers, n_classes):
    byc = {c: [] for c in range(n_classes)}
    for (y, x) in centers:
        c = int(labels[y, x])
        if 0 <= c < n_classes and (c not in IGNORE_LABELS):
            byc[c].append((y, x))
    return byc

def reserve_rare_centers(byc, seed):
    rng = np.random.default_rng(seed)
    val_res = []
    test_res = []
    used = set()

    for c in sorted(RARE_CLASSES):
        lst = byc.get(c, [])
        if not lst:
            continue

        lst = lst.copy()
        rng.shuffle(lst)

        n_val = min(MIN_VAL_PER_RARE, len(lst))
        val_part = lst[:n_val]

        remain = lst[n_val:]
        n_test = min(MIN_TEST_PER_RARE, len(remain))
        test_part = remain[:n_test]

        for pt in val_part:
            val_res.append(pt); used.add(pt)
        for pt in test_part:
            test_res.append(pt); used.add(pt)

    return val_res, test_res, used



# =============================
# Main
# =============================
def main():
    ensure_dirs(OUT_DIR)

    cube, cube_profile = read_cube(CUBE_TIF)
    labels, lab_profile = read_labels(LABEL_TIF)

    B, H, W = cube.shape
    if labels.shape != (H, W):
        raise RuntimeError(
            f"Label shape {labels.shape} != cube spatial shape {(H, W)}.\n"
            f"Your label must be aligned/resampled to the cube grid first."
        )

    n_classes, uniq = infer_classes(labels)
    if n_classes <= 1:
        raise RuntimeError(f"Found n_classes={n_classes}. Check label values / IGNORE_LABELS / LABEL_OFFSET.")

    print("Cube shape:", cube.shape)
    print("Label shape:", labels.shape)
    print("Unique label values (excluding ignore):", uniq)
    print("n_classes:", n_classes)

    centers = valid_centers(labels, PATCH_SIZE)
    print("Valid centers:", len(centers))

    # --------------------------
    # HYBRID SPLIT:
    #   1) reserve some samples of rare classes for VAL/TEST
    #   2) block split the rest
    # --------------------------
    byc = centers_by_class(labels, centers, n_classes)
    val_res, test_res, used = reserve_rare_centers(byc, SEED)

    centers_remaining = [pt for pt in centers if pt not in used]

    if USE_BLOCK_SPLIT:
        train_centers, val_centers, test_centers = block_split(centers_remaining, BLOCK_SIZE, SEED)
        split_mode = f"block_split(block={BLOCK_SIZE}) + rare_reserve(val={MIN_VAL_PER_RARE},test={MIN_TEST_PER_RARE})"
    else:
        train_centers, val_centers, test_centers = random_split(centers_remaining, SEED)
        split_mode = f"random_split + rare_reserve(val={MIN_VAL_PER_RARE},test={MIN_TEST_PER_RARE})"

    # add reserved rare samples
    val_centers.extend(val_res)
    test_centers.extend(test_res)

    rng = np.random.default_rng(SEED)
    rng.shuffle(train_centers)
    rng.shuffle(val_centers)
    rng.shuffle(test_centers)

    print("\nSplit mode:", split_mode)
    print("Train/Val/Test:", len(train_centers), len(val_centers), len(test_centers))
    print(f"Reserved for rare classes -> VAL: {len(val_res)} | TEST: {len(test_res)} | removed from split pool: {len(used)}")

    # Class distribution per split (critical sanity-check)
    train_cnt = split_class_counts(labels, train_centers, n_classes)
    val_cnt   = split_class_counts(labels, val_centers, n_classes)
    test_cnt  = split_class_counts(labels, test_centers, n_classes)

    print_counts("TRAIN", train_cnt)
    print_counts("VAL",   val_cnt)
    print_counts("TEST",  test_cnt)

    # Warning if any class disappears in val/test
    missing_val  = np.where(val_cnt == 0)[0].tolist()
    missing_test = np.where(test_cnt == 0)[0].tolist()
    if missing_val:
        print(f"\nWARNING: classes missing in VAL: {missing_val}")
    if missing_test:
        print(f"\nWARNING: classes missing in TEST: {missing_test}")

    # Save patches
    idx = 0
    idx = save_patches(cube, labels, train_centers, PATCH_SIZE, "train", idx, OUT_DIR)
    idx = save_patches(cube, labels, val_centers,   PATCH_SIZE, "val",   idx, OUT_DIR)
    idx = save_patches(cube, labels, test_centers,  PATCH_SIZE, "test",  idx, OUT_DIR)

    meta = {
        "source_cube_tif": CUBE_TIF,
        "source_label_tif": LABEL_TIF,
        "height": int(H),
        "width": int(W),
        "bands": int(B),
        "patch_size": int(PATCH_SIZE),
        "n_classes": int(n_classes),
        "ignore_labels": sorted(list(IGNORE_LABELS)),
        "label_offset": int(LABEL_OFFSET),
        "split_mode": split_mode,
        "splits": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
            "seed": SEED,
            "block_size": int(BLOCK_SIZE) if USE_BLOCK_SPLIT else None,
            "rare_classes": sorted(list(RARE_CLASSES)),
            "min_val_per_rare": int(MIN_VAL_PER_RARE),
            "min_test_per_rare": int(MIN_TEST_PER_RARE),
        },
        "counts": {
            "total_centers": int(len(centers)),
            "train_samples": int(len(train_centers)),
            "val_samples": int(len(val_centers)),
            "test_samples": int(len(test_centers)),
        },
        "class_counts": {
            "train": train_cnt.tolist(),
            "val": val_cnt.tolist(),
            "test": test_cnt.tolist(),
        },
        "raster_profile_cube": {
            "crs": str(cube_profile.get("crs")),
            "transform": str(cube_profile.get("transform")),
        },
        "raster_profile_label": {
            "crs": str(lab_profile.get("crs")),
            "transform": str(lab_profile.get("transform")),
        },
        "zero_to_nan": bool(ZERO_TO_NAN),
    }

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDONE.")
    print("OUT_DIR:", OUT_DIR)
    print("meta.json written.")

if __name__ == "__main__":
    main()
