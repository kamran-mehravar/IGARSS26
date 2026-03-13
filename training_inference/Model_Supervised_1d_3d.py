import os
import json
import time
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.notebook import tqdm


# =============================
# 1) CONFIGURATION
# =============================
class Config:
    # Dataset root that contains: meta.json, splits/train.txt, splits/val.txt, splits/test.txt
    data_dir = "/content/emit_cuprite_dataset_out"

    # Outputs: checkpoints, histories, curves, confusion matrices, maps
    out_dir = "/content/runs_A100_Final"

    # Training control
    do_train = True           # set False to skip training and only run inference/viz
    epochs_3d = 100
    epochs_1d = 100
    patience = 10

    # Batch sizes (A100-friendly defaults)
    batch_3d = 512
    batch_1d = 4096

    # Optimization
    lr = 1e-3
    weight_decay = 1e-4

    # Workers
    workers_train = os.cpu_count() or 2
    workers_infer = 2         # safe for inference in Colab

    # Mixed precision
    amp = True

    # Reproducibility
    seed = 42

    # Scene inference paths (override meta paths if you want)
    # If set to None, the code will try to use meta["source_cube_tif"] / meta["source_label_tif"]
    CUBE_PATH = "/content/drive/MyDrive/resize-continuum.tif"
    LABEL_PATH = "/content/drive/MyDrive/klabels10_georef.tif"

    # Map inference batch sizes
    map_bs_3d = 1024
    map_bs_1d = 16384

cfg = Config()


# =============================
# 2) SETUP & UTILS
# =============================
def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        # A100 speed boosts (safe on recent PyTorch)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Running on CPU")
    return device

DEVICE = setup_device()
AMP = bool(cfg.amp and DEVICE == "cuda")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdirp(p):
    os.makedirs(p, exist_ok=True)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def read_meta(root):
    with open(os.path.join(root, "meta.json"), "r") as f:
        return json.load(f)

def read_split(root, split):
    split_file = os.path.join(root, "splits", f"{split}.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        # IMPORTANT: convert Windows paths to Linux style
        return [line.strip().replace("\\", "/") for line in f if line.strip()]

def plot_cm(cm, title, path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    if cm.shape[0] < 20:
        thr = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                c = "white" if cm[i, j] > thr else "black"
                plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=c, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_curves(history, title, path):
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(ep, [h["train_loss"] for h in history], "r-", alpha=0.6, label="Tr Loss")
    ax1.plot(ep, [h["val_loss"] for h in history], "r--o", label="Val Loss")
    ax2.plot(ep, [h["val_acc"] for h in history], "b--s", label="Val Acc")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="r")
    ax2.set_ylabel("Acc", color="b")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)

    plt.savefig(path, dpi=150)
    plt.close()

def ensure_meta_dir(cfg_obj):
    """
    Consolidated meta.json discovery fix:
    If cfg.data_dir/meta.json doesn't exist, try /content/meta.json and switch data_dir.
    """
    meta_json_path = os.path.join(cfg_obj.data_dir, "meta.json")
    if os.path.exists(meta_json_path):
        return

    print(f"Warning: meta.json not found at {meta_json_path}.")
    alt_meta_json_path = os.path.join("/content", "meta.json")
    if os.path.exists(alt_meta_json_path):
        cfg_obj.data_dir = "/content"
        print(f"Adjusting data_dir to: {cfg_obj.data_dir}")
        return

    print("Meta.json still not found. Listing /content directory for diagnosis:")
    for root, dirs, files in os.walk("/content"):
        level = root.replace("/content", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for fn in files:
            print(f"{subindent}{fn}")
    raise FileNotFoundError(f"meta.json not found in expected paths: {meta_json_path} or {alt_meta_json_path}")


# =============================
# 3) DATASET
# =============================
class NPZPatchDataset(Dataset):
    def __init__(self, root, split, mode="3d"):
        self.root = root
        self.mode = mode
        self.paths = read_split(root, split)

        meta = read_meta(root)
        self.P = int(meta["patch_size"])
        self.c = self.P // 2
        self.B = int(meta["bands"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        fpath = os.path.join(self.root, rel)
        try:
            d = np.load(fpath)
            x = d["x"].astype(np.float32)
            y = int(d["y"])
        except Exception as e:
            print(f"Error loading {rel}: {e}")
            # return dummy with correct shape/bands
            if self.mode == "1d":
                return torch.zeros((self.B,), dtype=torch.float32), torch.tensor(0)
            return torch.zeros((self.B, self.P, self.P), dtype=torch.float32), torch.tensor(0)

        x = np.nan_to_num(x)

        if self.mode == "1d":
            s = x[:, self.c, self.c]
            med = np.median(s)
            mad = np.median(np.abs(s - med)) + 1e-6
            s = (s - med) / mad
            return torch.from_numpy(s.astype(np.float32)), torch.tensor(y)

        return torch.from_numpy(x), torch.tensor(y)


# =============================
# 4) MODELS (MUST MATCH FOR LOAD)
# =============================
class SpectralCNN1D(nn.Module):
    def __init__(self, bands, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))

class ResNet3D(nn.Module):
    def __init__(self, in_c, red_c, n_cls):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(in_c, red_c, 1, bias=False),
            nn.BatchNorm2d(red_c),
            nn.ReLU()
        )
        self.bb = resnet18(weights=None)
        self.bb.conv1 = nn.Conv2d(red_c, 64, 7, stride=2, padding=3, bias=False)
        self.bb.fc = nn.Linear(self.bb.fc.in_features, n_cls)

    def forward(self, x):
        return self.bb(self.red(x))


# =============================
# 5) TRAINING & EVAL
# =============================
class EarlyStopping:
    def __init__(self, patience=7, path="checkpoint.pt", min_delta=1e-3):
        self.patience = patience
        self.path = path
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def _autocast_ctx():
    return torch.amp.autocast("cuda", enabled=(AMP and DEVICE == "cuda"))

def train_epoch(model, ldr, opt, crit, scaler):
    model.train()
    sum_loss, corr, tot = 0.0, 0, 0

    for x, y in tqdm(ldr, desc="Tr", leave=False):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with _autocast_ctx():
            out = model(x)
            loss = crit(out, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        bs = y.numel()
        sum_loss += float(loss.item()) * bs
        corr += int((out.argmax(1) == y).sum().item())
        tot += bs

    return sum_loss / max(tot, 1), corr / max(tot, 1)

@torch.no_grad()
def eval_model(model, ldr, crit, n_cls):
    model.eval()
    sum_loss, tot = 0.0, 0
    yt, yp = [], []

    for x, y in tqdm(ldr, desc="Eval", leave=False):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with _autocast_ctx():
            out = model(x)
            loss = crit(out, y)

        bs = y.numel()
        sum_loss += float(loss.item()) * bs
        tot += bs

        yt.extend(y.detach().cpu().numpy().tolist())
        yp.extend(out.argmax(1).detach().cpu().numpy().tolist())

    yt = np.asarray(yt, dtype=np.int32)
    yp = np.asarray(yp, dtype=np.int32)

    cm = np.bincount(yt * n_cls + yp, minlength=n_cls**2).reshape(n_cls, n_cls)
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    ious = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm) + 1e-9)
    miou = float(np.mean(ious))

    return sum_loss / max(tot, 1), acc, miou, cm

def fit_smart(tag, model, tr_ldr, va_ldr, n_cls, w, epochs, out_dir):
    print(f"\nStart {tag} Training...")
    model.to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and DEVICE == "cuda"))
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3)

    save_path = os.path.join(out_dir, f"best_{tag}.pt")
    es = EarlyStopping(patience=cfg.patience, path=save_path)

    hist = []
    for ep in range(1, epochs + 1):
        tl, ta = train_epoch(model, tr_ldr, opt, crit, scaler)
        vl, va, viou, _ = eval_model(model, va_ldr, crit, n_cls)

        print(f"[{tag}] Ep {ep:03d} | TrL: {tl:.4f} TrA: {ta:.4f} | VaL: {vl:.4f} VaA: {va:.4f} mIoU: {viou:.4f}")
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_acc": va})

        sch.step(vl)
        es(vl, model)
        if es.early_stop:
            print(f"Early Stopping at Epoch {ep}")
            break

    # Load best
    state = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(state)
    save_json(os.path.join(out_dir, f"history_{tag}.json"), hist)
    return model, hist


# =============================
# 6) MAP INFERENCE (UPDATED + CONSOLIDATED)
# =============================
def generate_map(model, cube, labels, P, ignore, bs, mode, num_workers):
    """
    Consolidated version:
    - non_blocking GPU copies
    - pin_memory only if CUDA
    - persistent_workers only if num_workers>0
    - AMP enabled only when CUDA exists
    - safe numpy indexing for out_map assignment
    """
    model.eval()
    H, W = labels.shape
    r = P // 2

    mask = np.ones_like(labels, dtype=bool)
    mask[:r, :] = 0
    mask[-r:, :] = 0
    mask[:, :r] = 0
    mask[:, -r:] = 0
    for i in ignore:
        mask[labels == i] = 0

    coords = list(zip(*np.where(mask)))
    out_map = np.full((H, W), -1, dtype=np.int32)

    class InfDS(Dataset):
        def __len__(self):
            return len(coords)

        def __getitem__(self, i):
            y, x = coords[i]
            if mode == "3d":
                patch = cube[:, y - r : y + r + 1, x - r : x + r + 1]
                return torch.from_numpy(patch), y, x

            s = cube[:, y, x]
            med = np.median(s)
            mad = np.median(np.abs(s - med)) + 1e-6
            s = (s - med) / mad
            return torch.from_numpy(s.astype(np.float32)), y, x

    ldr = DataLoader(
        InfDS(),
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    print(f"Generating {mode} map ({len(coords)} pixels)...")
    with torch.no_grad():
        for bx, by, bc in tqdm(ldr, desc="Map", leave=False):
            bx = bx.to(DEVICE, non_blocking=True)
            with _autocast_ctx():
                p = model(bx).argmax(1).detach().cpu().numpy()

            # robust assignment (avoid relying on .numpy() for list/int)
            out_map[np.asarray(by), np.asarray(bc)] = p

    return out_map


# =============================
# 7) VISUALIZATION UTILS (UPDATED: YOUR CUSTOM COLORS)
# =============================
def save_colored_map(data, out_path, title, cmap_type="classes", num_classes=10):
    """
    Saves a 2D numpy array as a colored PNG.
    - data < 0 is treated as background and masked (white).
    - cmap_type:
        'classes' -> discrete class colors [0..K-1] (UPDATED palette to match your reference)
        'diff'    -> 0 green (match), 1 red (mismatch)
    """
    plt.figure(figsize=(12, 10))
    masked_data = np.ma.masked_where(data < 0, data)

    # Make background white for masked pixels
    ax = plt.gca()
    ax.set_facecolor("white")

    if cmap_type == "classes":
       
        colors = [
            "#1f77b4",  # 0: Blue
            "#ff7f0e",  # 1: Orange
            "#2ca02c",  # 2: Green
            "#d62728",  # 3: Red
            "#9467bd",  # 4: Purple
            "#8c564b",  # 5: Brown
            "#e377c2",  # 6: Pink
            "#7f7f7f",  # 7: Gray
            "#bcbd22",  # 8: Olive
            "#17becf",  # 9: Cyan
        ]
        cmap = mcolors.ListedColormap(colors[:num_classes])
        cmap.set_bad(color="white")

        bounds = np.arange(num_classes + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = plt.imshow(masked_data, cmap=cmap, norm=norm, interpolation="nearest")
        cbar = plt.colorbar(im, ticks=np.arange(num_classes))
        cbar.ax.set_ylabel("Class ID", rotation=270, labelpad=15)

    elif cmap_type == "diff":
        # 0=Match (Green), 1=Mismatch (Red)
        cmap = mcolors.ListedColormap(["#2ca02c", "#d62728"])
        cmap.set_bad(color="white")

        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = plt.imshow(masked_data, cmap=cmap, norm=norm, interpolation="nearest")
        cbar = plt.colorbar(im, ticks=[0, 1])
        cbar.ax.set_yticklabels(["Match", "Mismatch"])

    else:
        raise ValueError(f"Unknown cmap_type: {cmap_type}")

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_1d_vs_3d_matrix(map1, map3, num_classes, out_path):
    """
    Agreement/confusion matrix comparing 1D predictions (Rows) vs 3D predictions (Cols).
    Only considers pixels where both maps have valid predictions (>=0).
    """
    print("Calculating 1D vs 3D Confusion Matrix...")

    valid_mask = (map1 >= 0) & (map3 >= 0)
    p1 = map1[valid_mask].astype(np.int32)
    p3 = map3[valid_mask].astype(np.int32)

    if p1.size == 0:
        print("Warning: No overlapping valid pixels found for matrix comparison.")
        return

    idx = p1 * num_classes + p3
    counts = np.bincount(idx, minlength=num_classes**2)
    cm = counts.reshape(num_classes, num_classes)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Oranges", interpolation="nearest")
    plt.title("Agreement Matrix: 1D Model (Rows) vs 3D Model (Cols)", fontsize=14)
    plt.xlabel("3D Model Predictions", fontsize=12)
    plt.ylabel("1D Model Predictions", fontsize=12)
    plt.colorbar()

    thr = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            txt = f"{val}" if val > 0 else "."
            color = "white" if val > thr else "black"
            plt.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Matrix saved to {out_path}")


# =============================
# 8) MAIN ORCHESTRATOR
# =============================
def main():
    set_seed(cfg.seed)
    mkdirp(cfg.out_dir)

    # Ensure meta.json is discoverable (consolidated from your fix)
    ensure_meta_dir(cfg)

    meta = read_meta(cfg.data_dir)
    B = int(meta["bands"])
    K = int(meta["n_classes"])
    P = int(meta["patch_size"])
    ign = set(meta.get("ignore_labels", []))

    # Class weights from meta
    cw = np.asarray(meta["class_counts"]["train"], dtype=np.float32)
    w = torch.tensor(1.0 / np.sqrt(np.maximum(cw, 1.0)), dtype=torch.float32, device=DEVICE)
    w = w / w.mean()

    # -------------------------
    # TRAINING (OPTIONAL)
    # -------------------------
    best_3d_path = os.path.join(cfg.out_dir, "best_3D.pt")
    best_1d_path = os.path.join(cfg.out_dir, "best_1D.pt")

    if cfg.do_train:
        # training dataloaders use cfg.workers_train and persistent_workers=True
        kw_train = {
            "num_workers": cfg.workers_train,
            "pin_memory": (DEVICE == "cuda"),
            "persistent_workers": (cfg.workers_train > 0),
        }

        tr3 = DataLoader(NPZPatchDataset(cfg.data_dir, "train", "3d"),
                         batch_size=cfg.batch_3d, shuffle=True, **kw_train)
        va3 = DataLoader(NPZPatchDataset(cfg.data_dir, "val", "3d"),
                         batch_size=cfg.batch_3d, shuffle=False, **kw_train)
        te3 = DataLoader(NPZPatchDataset(cfg.data_dir, "test", "3d"),
                         batch_size=cfg.batch_3d, shuffle=False, **kw_train)

        tr1 = DataLoader(NPZPatchDataset(cfg.data_dir, "train", "1d"),
                         batch_size=cfg.batch_1d, shuffle=True, **kw_train)
        va1 = DataLoader(NPZPatchDataset(cfg.data_dir, "val", "1d"),
                         batch_size=cfg.batch_1d, shuffle=False, **kw_train)
        te1 = DataLoader(NPZPatchDataset(cfg.data_dir, "test", "1d"),
                         batch_size=cfg.batch_1d, shuffle=False, **kw_train)

        # Train 3D
        m3, h3 = fit_smart("3D", ResNet3D(B, 16, K), tr3, va3, K, w, cfg.epochs_3d, cfg.out_dir)
        plot_curves(h3, "3D Training", os.path.join(cfg.out_dir, "curve_3d.png"))

        # Train 1D
        m1, h1 = fit_smart("1D", SpectralCNN1D(B, K), tr1, va1, K, w, cfg.epochs_1d, cfg.out_dir)
        plot_curves(h1, "1D Training", os.path.join(cfg.out_dir, "curve_1d.png"))

        # Test evaluation
        print("\nRunning Test Evaluation...")
        crit = nn.CrossEntropyLoss(weight=w)

        _, acc3, iou3, cm3 = eval_model(m3, te3, crit, K)
        print(f"TEST 3D -> Acc: {acc3:.4f}, mIoU: {iou3:.4f}")
        plot_cm(cm3, "Test Confusion Matrix (3D)", os.path.join(cfg.out_dir, "cm_3d_test.png"))

        _, acc1, iou1, cm1 = eval_model(m1, te1, crit, K)
        print(f"TEST 1D -> Acc: {acc1:.4f}, mIoU: {iou1:.4f}")
        plot_cm(cm1, "Test Confusion Matrix (1D)", os.path.join(cfg.out_dir, "cm_1d_test.png"))

        save_json(os.path.join(cfg.out_dir, "test_summary.json"), {
            "test_3d": {"acc": float(acc3), "mIoU": float(iou3)},
            "test_1d": {"acc": float(acc1), "mIoU": float(iou1)},
        })

    else:
        print("Skipping training (cfg.do_train=False).")

    # -------------------------
    # INFERENCE + MAPS + VIZ
    # -------------------------
    if not (os.path.exists(best_3d_path) and os.path.exists(best_1d_path)):
        print(f"ERROR: Model files not found in {cfg.out_dir}")
        print("If runtime restarted and files were cleared, set cfg.do_train=True and retrain.")
        return

    print("\nLoading saved models...")
    model3d = ResNet3D(B, 16, K).to(DEVICE)
    model3d.load_state_dict(torch.load(best_3d_path, map_location=DEVICE))
    print("3D model loaded.")

    model1d = SpectralCNN1D(B, K).to(DEVICE)
    model1d.load_state_dict(torch.load(best_1d_path, map_location=DEVICE))
    print("1D model loaded.")

    # Resolve scene paths
    cube_path = cfg.CUBE_PATH or meta.get("source_cube_tif", None)
    label_path = cfg.LABEL_PATH or meta.get("source_label_tif", None)

    if not cube_path or not label_path:
        print("ERROR: Scene paths are not defined. Set cfg.CUBE_PATH/LABEL_PATH or ensure meta provides them.")
        return

    if not (os.path.exists(cube_path) and os.path.exists(label_path)):
        print("ERROR: TIF files not found. Check cfg.CUBE_PATH / cfg.LABEL_PATH.")
        print(f"Cube:  {cube_path}")
        print(f"Label: {label_path}")
        return

    print("\nLoading full-scene data...")
    with rasterio.open(cube_path) as s:
        cube = s.read().astype(np.float32)
        prof = s.profile
    with rasterio.open(label_path) as s:
        lbl = s.read(1).astype(np.int32)

    print("\nGenerating prediction maps...")
    map3 = generate_map(model3d, cube, lbl, P, ign, cfg.map_bs_3d, "3d", cfg.workers_infer)
    map1 = generate_map(model1d, cube, lbl, P, ign, cfg.map_bs_1d, "1d", cfg.workers_infer)

    print("Computing difference map...")
    diff = np.full_like(map3, -1)
    valid = (map3 >= 0) & (map1 >= 0)
    diff[valid] = (map3[valid] != map1[valid]).astype(np.int32)

    # Save outputs
    path_map = os.path.join(cfg.out_dir, "maps")
    mkdirp(path_map)

    prof.update(count=1, dtype=rasterio.int32)
    print("\nSaving GeoTIFFs...")
    with rasterio.open(os.path.join(path_map, "pred_3d.tif"), "w", **prof) as d:
        d.write(map3, 1)
    with rasterio.open(os.path.join(path_map, "pred_1d.tif"), "w", **prof) as d:
        d.write(map1, 1)
    with rasterio.open(os.path.join(path_map, "diff_1d_3d.tif"), "w", **prof) as d:
        d.write(diff, 1)

    # Quick diff PNG (basic)
    plt.figure(figsize=(10, 6))
    plt.imshow(diff)
    plt.title("Difference Map (0=Same, 1=Diff)")
    plt.colorbar()
    plt.savefig(os.path.join(path_map, "diff_map.png"), dpi=150)
    plt.close()

    # Colored visual maps (custom palette + diff palette)
    print("Saving visualization PNGs...")
    save_colored_map(map3, os.path.join(path_map, "visual_map_3d.png"), f"3D Prediction Map ({K} Classes)", "classes", K)
    save_colored_map(map1, os.path.join(path_map, "visual_map_1d.png"), f"1D Prediction Map ({K} Classes)", "classes", K)
    save_colored_map(diff, os.path.join(path_map, "visual_diff_map.png"), "Difference Map (Match vs Mismatch)", "diff", K)

    # 1D vs 3D matrix
    plot_1d_vs_3d_matrix(map1, map3, K, os.path.join(path_map, "matrix_1d_vs_3d.png"))

    print(f"\nALL DONE. Outputs saved to: {cfg.out_dir}")
    print(f"Maps folder: {path_map}")


if __name__ == "__main__":
    main()
