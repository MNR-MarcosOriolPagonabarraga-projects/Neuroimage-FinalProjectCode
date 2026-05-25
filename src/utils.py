import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subject_stratified_split(records: list, val_fraction: float, seed: int,) -> tuple[list[int], list[int]]:
    """Return train/val *window* indices with no subject appearing in both splits."""
    sid_to_label: dict[str, int] = {}
    for rec in records:
        sid_to_label.setdefault(rec.subject_id, rec.label)

    subjects = sorted(set(sid_to_label))
    labels = [sid_to_label[s] for s in subjects]

    if len(subjects) < 2:
        raise RuntimeError(
            f"Need subjects from at least 2 participants; got {len(subjects)} usable subject(s)."
        )
    try:
        train_subj, val_subj = train_test_split(
            subjects,
            test_size=val_fraction,
            random_state=seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )
    except ValueError as e:
        raise RuntimeError(
            "Stratified subject split failed — need more samples per class. "
            "Try smaller val_fraction or merge more data."
        ) from e

    train_set, val_set = set(train_subj), set(val_subj)
    train_idx = [i for i, r in enumerate(records) if r.subject_id in train_set]
    val_idx = [i for i, r in enumerate(records) if r.subject_id in val_set]
    return train_idx, val_idx