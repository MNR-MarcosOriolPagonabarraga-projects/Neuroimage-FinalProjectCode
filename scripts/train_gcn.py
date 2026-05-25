import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.fmri_dataset import CHUNK_LENGTH, FmriRestChunkDataset, build_window_records, discover_feature_paths
from src.networks import fMRIGCN
from src.utils import _set_seeds, _subject_stratified_split

PROCESSED_DATA_PATH = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG_SEED = 42
EPOCHS = 40
BATCH_SIZE = 32
LR = 5e-4 # Un poco más alto para GCN
WEIGHT_DECAY = 1e-3
VAL_FRACTION = 0.2
EARLY_STOP_PATIENCE = 8


def _run_epoch(model, loader, optimizer: torch.optim.Optimizer | None):
    train = optimizer is not None
    model.train(train)
    losses, correct, total = [], 0, 0

    for batch in loader:
        y = batch["label"].to(DEVICE)
        x_act = batch["activation"].to(DEVICE)
        x_adj = batch["corr"].to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x_act, x_adj)
        loss = F.cross_entropy(logits, y)
        
        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return np.mean(losses), correct / max(total, 1)

def main():
    _set_seeds(RNG_SEED)
    paths = discover_feature_paths(PROCESSED_DATA_PATH)
    records, meta = build_window_records(paths)

    if not records or meta["n_rois"] is None:
        raise SystemExit(f"No hay datos bajo {PROCESSED_DATA_PATH!r}.")

    train_idx, val_idx = _subject_stratified_split(records, VAL_FRACTION, RNG_SEED)
    train_recs = [records[i] for i in train_idx]
    val_recs = [records[i] for i in val_idx]

    # Pedimos modo "both" al Dataset para que nos devuelva activation y corr
    train_loader = DataLoader(FmriRestChunkDataset(train_recs, "both"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FmriRestChunkDataset(val_recs, "both"), batch_size=BATCH_SIZE, shuffle=False)

    n_rois = int(meta["n_rois"])
    model = fMRIGCN(n_rois).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, stale = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, opt)
        with torch.no_grad():
            va_loss, va_acc = _run_epoch(model, val_loader, None)

        print(f"Epoch {epoch:03d} | Train Loss {tr_loss:.4f} Acc {tr_acc:.3f} | Val Loss {va_loss:.4f} Acc {va_acc:.3f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            stale = 0
            torch.save(model.state_dict(), "outputs/best_gcn.pt")
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()