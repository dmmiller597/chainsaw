"""Training data: chain-id selection and the pairwise feature/label dataset.

Preprocessing writes one ``.npz`` per chain into a feature directory and a label
directory (both keyed by chain id). Each feature file holds an ``(C, L, L)`` array
under ``arr_0`` (the channels the model consumes); each label file holds the
``(L, L)`` pairwise co-membership matrix under ``arr_0``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_test_ids(label_path: str | Path, feature_path: str | Path, csv_path: str | Path | None = None) -> list[str]:
    """Chain ids present in both the label and feature directories (optionally CSV-filtered)."""
    label_ids = {path.stem for path in Path(label_path).glob("*.npz") if path.is_file()}
    feature_ids = {path.stem for path in Path(feature_path).glob("*.npz") if path.is_file()}
    ids = label_ids & feature_ids
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        ids &= set(df["chain_id"].astype(str))
    return sorted(ids)


def filter_plddt(df_path: str | Path, ids: list[str], threshold: int = 90) -> list[str]:
    """Keep ids whose pLDDT exceeds ``threshold`` according to the CSV at ``df_path``."""
    df = pd.read_csv(df_path)
    retained = set(df.loc[df["plddt"] > threshold, "casp_id"].astype(str))
    return [chain_id for chain_id in ids if chain_id in retained]


class PairwiseFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Serve ``(features, labels)`` tensors from preprocessed per-chain ``.npz`` files.

    Args:
        feature_dir: Directory of ``{chain_id}.npz`` feature files, each ``(C, L, L)``.
        label_dir: Directory of ``{chain_id}.npz`` pairwise label files, each ``(L, L)``.
        chain_ids: Chains to serve (e.g. from :func:`get_test_ids`).
    """

    def __init__(self, feature_dir: str | Path, label_dir: str | Path, chain_ids: list[str]) -> None:
        self.feature_dir = Path(feature_dir)
        self.label_dir = Path(label_dir)
        self.chain_ids = tuple(chain_ids)

    def __len__(self) -> int:
        return len(self.chain_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chain_id = self.chain_ids[index]
        with np.load(self.feature_dir / f"{chain_id}.npz") as feature_archive:
            features = feature_archive["arr_0"]  # (C, L, L)
        with np.load(self.label_dir / f"{chain_id}.npz") as label_archive:
            labels = label_archive["arr_0"]  # (L, L)
        if features.ndim != 3:
            raise ValueError(f"Expected 3-D features for '{chain_id}', got shape {features.shape}")
        if labels.ndim != 2 or labels.shape != features.shape[-2:]:
            raise ValueError(
                f"Feature/label shape mismatch for '{chain_id}': features {features.shape}, labels {labels.shape}"
            )
        return torch.from_numpy(features).float(), torch.from_numpy(labels).float()
