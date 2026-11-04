"""Data acquisition and global-time splits. Raw GroupLens data stays local."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
# SHA-256 measured from the official archive downloaded for the committed run.
EXPECTED_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
DTYPE = [("user", "i8"), ("item", "i8"), ("rating", "i8"), ("timestamp", "i8")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(directory: Path) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "ml-100k.zip"
    if not archive.exists():
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "TemporalRecommendations/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(10_000_001)
        if len(payload) > 10_000_000:
            raise ValueError("Dataset response exceeds expected size")
        archive.write_bytes(payload)
    digest = sha256(archive)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Dataset digest mismatch: {digest}")
    # Extract exactly one known member, never arbitrary paths from an archive.
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as handle:
        payload = handle.read("ml-100k/u.data")
    target = directory / "u.data"
    target.write_bytes(payload)
    return target, digest


def load_ratings(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if values.shape[1] != 4:
        raise ValueError("Expected user, item, rating, Unix timestamp")
    if np.any(values[:, :2] < 1) or np.any((values[:, 2] < 1) | (values[:, 2] > 5)):
        raise ValueError("Invalid identifier or rating")
    if np.any(values[:, 3] < 0):
        raise ValueError("Negative timestamp")
    if len(np.unique(values[:, :2], axis=0)) != len(values):
        raise ValueError("Duplicate user-item ratings need an explicit update policy")
    rows = np.array([tuple(row) for row in values], dtype=DTYPE)
    return np.sort(rows, order=["timestamp", "user", "item"])


def temporal_split(rows: np.ndarray, train_fraction: float = .70, validation_end: float = .85):
    """Cut by global time; all equal timestamps remain in the same partition."""
    if len(rows) < 10 or not 0 < train_fraction < validation_end < 1:
        raise ValueError("Insufficient rows or invalid split fractions")
    ordered = np.sort(rows, order=["timestamp", "user", "item"])
    first = int(ordered["timestamp"][int(len(rows) * train_fraction) - 1])
    second = int(ordered["timestamp"][int(len(rows) * validation_end) - 1])
    train = ordered[ordered["timestamp"] <= first]
    validation = ordered[(ordered["timestamp"] > first) & (ordered["timestamp"] <= second)]
    test = ordered[ordered["timestamp"] > second]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("Tied timestamps produced an empty temporal partition")
    return train, validation, test, {"train_end": first, "validation_end": second}


def history(rows: np.ndarray) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for row in rows:
        out.setdefault(int(row["user"]), set()).add(int(row["item"]))
    return out
