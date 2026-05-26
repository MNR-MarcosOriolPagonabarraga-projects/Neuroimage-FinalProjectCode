import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import CHUNK_LENGTH



@dataclass(frozen=True)
class WindowRecord:
    subject_id: str
    activation: np.ndarray  # (CHUNK_LENGTH, n_rois) float32
    corr_matrix: np.ndarray  # (n_rois, n_rois) float32
    label: int  # 0 young, 1 adult


class FmriRestChunkDataset(Dataset):
    """
    Expands each scan into non-overlapping 20-frame windows (stride ``CHUNK_LENGTH``).
    Trailing segments shorter than 20 frames are dropped.
    """

    def __init__(
        self,
        records: list[WindowRecord],
        mode: str,
    ):
        """
        Args:
            records: Pre-built window rows (train or val slice).
            mode: ``"activation"`` | ``"corr"`` | ``"both"`` → what tensors are returned.
        """
        if mode not in ("activation", "corr", "both"):
            raise ValueError(f"Unknown mode={mode}")
        self._records = records
        self._mode = mode

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        rec = self._records[idx]
        # Z-score within window over time (per ROI): stabilizes amplitude across scanners.
        w = torch.from_numpy(rec.activation.astype(np.float32, copy=False))
        w = (w - w.mean(dim=0, keepdim=True)) / (w.std(dim=0, keepdim=True) + 1e-6)
        corr = torch.from_numpy(rec.corr_matrix.astype(np.float32, copy=False))

        label = torch.tensor(rec.label, dtype=torch.long)

        if self._mode == "activation":
            return {"activation": w.unsqueeze(0), "label": label}  # (1, T, R)
        if self._mode == "corr":
            return {"corr": corr.unsqueeze(0), "label": label}  # (1, R, R)
        return {
            "activation": w.unsqueeze(0),
            "corr": corr.unsqueeze(0),
            "label": label,
        }


def _subject_id_from_npz_filename(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_features.npz"):
        base = base[: -len("_features.npz")]
    return base


def discover_feature_paths(processed_root: str) -> list[str]:
    paths = []
    if not processed_root:
        return paths
    for fn in sorted(os.listdir(processed_root)):
        if not fn.endswith("_features.npz"):
            continue
        p = os.path.join(processed_root, fn)
        if os.path.isfile(p):
            paths.append(p)
    return paths


def build_window_records(paths: list[str]) -> tuple[list[WindowRecord], dict]:
    """Load all bundles; validate shapes; flatten into window records."""
    skipped_label: list[str] = []
    skipped_short: list[tuple[str, int]] = []
    skipped_shape: list[tuple[str, str]] = []
    records: list[WindowRecord] = []
    n_rois_ref: int | None = None
    n_files = 0

    for fp in paths:
        subject_id = _subject_id_from_npz_filename(fp)

        with np.load(fp, allow_pickle=False) as z:
            activation = np.asarray(z["activation_time_series"], dtype=np.float32)
            corr = np.asarray(z["corr_matrix"], dtype=np.float32)
            label_arr = np.asarray(z["label"])
            label_scalar = int(np.reshape(label_arr, (-1))[0])

        n_files += 1

        if label_scalar < 0:
            skipped_label.append(subject_id)
            continue

        if activation.ndim != 2:
            skipped_shape.append((subject_id, f"activation_rank_{activation.ndim}"))
            continue
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            skipped_shape.append((subject_id, f"corr_bad_shape_{corr.shape}"))
            continue

        T, R = activation.shape
        if n_rois_ref is None:
            n_rois_ref = R
        elif R != n_rois_ref:
            skipped_shape.append((subject_id, f"roi_mismatch_{R}_vs_{n_rois_ref}"))
            continue

        if corr.shape[0] != R:
            skipped_shape.append((subject_id, "corr_roi_mismatch"))
            continue

        if T < CHUNK_LENGTH:
            skipped_short.append((subject_id, T))
            continue

        stride = CHUNK_LENGTH
        for start in range(0, T - CHUNK_LENGTH + 1, stride):
            slice_ts = activation[start : start + CHUNK_LENGTH].copy()
            corr_copy = corr.copy()
            records.append(
                WindowRecord(
                    subject_id=subject_id,
                    activation=slice_ts,
                    corr_matrix=corr_copy,
                    label=label_scalar,
                )
            )

    meta = {
        "paths_used": len(paths),
        "files_opened_ok": n_files,
        "n_windows": len(records),
        "n_rois": n_rois_ref,
        "skipped_label": skipped_label,
        "skipped_short": skipped_short,
        "skipped_shape": skipped_shape,
    }
    return records, meta
