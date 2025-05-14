# train_ssd.py

import os
import yaml
from PIL import Image

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models.detection import ssd300_vgg16

# ─── 1. CONFIG ─────────────────────────────────────────────────────────────────

# Path to your data.yaml
YAML_CONFIG = "data.yaml"

# Hyperparameters
IMG_SIZE    = 300           # SSD300
NUM_EPOCHS  = 25
BATCH_SIZE  = 16
LR          = 1e-4          # start lower to avoid explosions
OUTPUT_DIR  = "checkpoints"

# ─── 2. DATASET WRAPPER ──────────────────────────────────────────────────────────

class YoloDetectionDataset(Dataset):
    """
    Expects this layout under `split_dir`:
      ├── images/    (*.jpg, *.png, etc.)
      └── labels/    (*.txt in YOLO format: cls cx cy w h, normalized)

    Filters out any image with 0 labels.
    """
    def __init__(self, split_dir, img_size=IMG_SIZE):
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            raise FileNotFoundError(f"Expected images/ and labels/ in {split_dir}")

        self.img_size = img_size
        self.samples = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg","jpeg","png")):
                continue
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            if not os.path.isfile(lbl_path):
                continue
            # read label lines
            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue  # skip images with no boxes
            self.samples.append((os.path.join(img_dir, fname), lbl_path))

        if not self.samples:
            raise RuntimeError(f"No valid training samples found in {split_dir}")

        # Tensor conversion & normalization (no resize here)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # parse YOLO labels
        boxes = []
        labels = []
        with open(lbl_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.split())
                # to absolute coords on original image
                cx, cy, bw, bh = cx*orig_w, cy*orig_h, bw*orig_w, bh*orig_h
                xmin = cx - bw/2; ymin = cy - bh/2
                xmax = cx + bw/2; ymax = cy + bh/2
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls)+1)  # SSD: 0=background

        # resize image & scale boxes
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y

        # to tensor + normalize
        img = self.to_tensor(img)
        img = self.normalize(img)

        target = {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}
        return img, target

def collate_fn(batch):
    imgs, targs = zip(*batch)
    return list(imgs), list(targs)


# ─── 3. TRAIN & VALIDATION ──────────────────────────────────────────────────────

def main():
    # load config
    with open(YAML_CONFIG) as f:
        cfg = yaml.safe_load(f)
    root      = cfg["path"]
    train_dir = os.path.join(root, cfg["train"])
    val_dir   = os.path.join(root, cfg["val"])
    test_rel  = cfg.get("test", None)
    test_dir  = os.path.join(root, test_rel) if test_rel else None
    num_cls   = cfg["nc"]

    # check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    # datasets + loaders
    train_ds = YoloDetectionDataset(train_dir)
    val_ds   = YoloDetectionDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # build SSD300 **offline**
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=None,
        num_classes=num_cls+1
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=12, gamma=0.1)

    best_val = float("inf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS+1):
        # — train —
        model.train()
        total_train = 0.0
        for imgs, targets in train_loader:
            imgs    = [img.to(device) for img in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            losses = model(imgs, targets)
            loss   = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += loss.item()

        scheduler.step()
        avg_train = total_train / len(train_loader)

        # — validate (keep in train mode under no_grad to get losses) —
        val_loss = 0.0
        with torch.no_grad():
            model.train()
            for imgs, targets in val_loader:
                imgs    = [img.to(device) for img in imgs]
                targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
                losses = model(imgs, targets)
                val_loss += sum(losses.values()).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch}/{NUM_EPOCHS} — Train: {avg_train:.4f} — Val: {avg_val:.4f}")

        # save best
        if avg_val < best_val:
            best_val = avg_val
            path = os.path.join(OUTPUT_DIR, "ssd_best.pth")
            torch.save(model.state_dict(), path)
            print(f"→ Saved best model: {path}")

    # — optional test —
    if test_dir:
        test_ds = YoloDetectionDataset(test_dir)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True, collate_fn=collate_fn)

        test_loss = 0.0
        with torch.no_grad():
            model.train()
            for imgs, targets in test_loader:
                imgs    = [img.to(device) for img in imgs]
                targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
                losses = model(imgs, targets)
                test_loss += sum(losses.values()).item()
        print(f"\nFinal Test Loss: {test_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    main()
