#!/usr/bin/env python
"""Preprocess FMNIST_TRAIN.csv and FMNIST_TEST.csv into a validated NPZ bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LABEL_CANDIDATES = ["label", "Label", "Class", "target", "Target", "y", "Y"]
PIXELS_CANDIDATES = ["pixels", "Pixels", "pixel_values", "Pixel_Values"]


def find_image_root(sample_image: str, search_root: Path) -> Path | None:
    candidates = [
        search_root,
        search_root / "train",
        search_root / "test",
        search_root / "images",
        search_root / "data",
        search_root / "FMNIST_TRAIN",
        search_root / "FMNIST_TEST",
    ]
    for root in candidates:
        if (root / sample_image).exists():
            return root

    for root, dirs, files in os.walk(search_root, topdown=True):
        dirs[:] = [d for d in dirs if d not in {"data_cache", "__pycache__", ".git", ".venv", "venv"}]
        if sample_image in files:
            return Path(root)
    return None


def load_manifest(df: pd.DataFrame, split_name: str, search_root: Path, image_root_hint: Path | None = None):
    required = {"Image_File", "Class"}
    if not required.issubset(df.columns):
        raise ValueError(f"{split_name}: expected columns {required}, found {set(df.columns)}")

    image_files = df["Image_File"].astype(str).tolist()
    y = df["Class"].astype(int).to_numpy()

    image_root = image_root_hint or find_image_root(image_files[0], search_root)
    if image_root is None:
        raise FileNotFoundError(
            f"{split_name}: could not locate image files. Ensure PNG files referenced in Image_File exist."
        )

    x = np.empty((len(image_files), 28, 28), dtype=np.uint8)
    for idx, image_name in enumerate(image_files):
        image_path = image_root / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"{split_name}: missing image file {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("L")
            if image.size != (28, 28):
                image = image.resize((28, 28))
            x[idx] = np.asarray(image, dtype=np.uint8)

    return x, y, image_root


def load_flat_pixel(df: pd.DataFrame, split_name: str):
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None and df.shape[1] == 785:
        label_col = df.columns[0]

    if label_col is None:
        raise ValueError(f"{split_name}: unable to infer label column for flat-pixel format")

    y = df[label_col].astype(int).to_numpy()

    pixels_col = next((c for c in PIXELS_CANDIDATES if c in df.columns), None)
    if pixels_col is not None:
        # Support packed-pixels format ("0 0 12 ...", 784 values per row)
        pixel_rows = (
            df[pixels_col]
            .astype(str)
            .str.replace(",", " ", regex=False)
            .str.split()
            .map(lambda row: [int(v) for v in row])
        )
        x = np.asarray(pixel_rows.tolist(), dtype=np.uint8)
    else:
        candidate_cols = [c for c in df.columns if c != label_col]
        drop_noise = {"Unnamed: 0", "id", "ID", "Image_File", "image_file", "filename", "file_name"}
        candidate_cols = [c for c in candidate_cols if c not in drop_noise]

        numeric_cols = []
        for col in candidate_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().all():
                numeric_cols.append(col)

        x = df[numeric_cols].to_numpy(dtype=np.float32)

    if x.shape[1] != 784:
        raise ValueError(f"{split_name}: expected 784 pixel values, found {x.shape[1]}")

    x = np.clip(x, 0, 255).reshape(-1, 28, 28).astype(np.uint8)
    return x, y


def _resolve_csv_paths(train_csv: Path, test_csv: Path):
    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv

    # Accept common lowercase variants without requiring manual renaming.
    fallback_pairs = [
        (Path("fashion-mnist_train.csv"), Path("fashion-mnist_test.csv")),
        (Path("fmnist_train.csv"), Path("fmnist_test.csv")),
    ]
    for train_fallback, test_fallback in fallback_pairs:
        if train_fallback.exists() and test_fallback.exists():
            return train_fallback, test_fallback

    raise FileNotFoundError(
        "Could not find dataset CSVs. Expected FMNIST_TRAIN.csv/FMNIST_TEST.csv or "
        "fashion-mnist_train.csv/fashion-mnist_test.csv in the project root."
    )


def preprocess_dataset(train_csv: Path, test_csv: Path, search_root: Path):
    train_csv, test_csv = _resolve_csv_paths(train_csv, test_csv)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    flat_error = None
    manifest_error = None

    try:
        x_train, y_train = load_flat_pixel(train_df, "train")
        x_test, y_test = load_flat_pixel(test_df, "test")
        source_type = "flat_pixel"
        source_detail = "inline 784-pixel representation"
    except Exception as exc:
        flat_error = str(exc)
        try:
            x_train, y_train, image_root = load_manifest(train_df, "train", search_root)
            x_test, y_test, _ = load_manifest(test_df, "test", search_root, image_root_hint=image_root)
            source_type = "manifest"
            source_detail = str(image_root)
        except Exception as exc2:
            manifest_error = str(exc2)
            raise ValueError(
                "Failed to parse FMNIST CSV files as either flat-pixel or manifest format.\n"
                f"Flat-pixel error: {flat_error}\n"
                f"Manifest error: {manifest_error}"
            ) from exc2

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "source_type": source_type,
        "source_detail": source_detail,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess FMNIST CSV files into processed/fmnist_preprocessed.npz")
    parser.add_argument("--train-csv", default="FMNIST_TRAIN.csv", type=Path)
    parser.add_argument("--test-csv", default="FMNIST_TEST.csv", type=Path)
    parser.add_argument("--search-root", default=".", type=Path)
    parser.add_argument("--out-dir", default="processed", type=Path)
    args = parser.parse_args()

    result = preprocess_dataset(args.train_csv, args.test_csv, args.search_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "fmnist_preprocessed.npz"
    meta_path = args.out_dir / "fmnist_preprocessed_meta.json"

    np.savez_compressed(
        npz_path,
        x_train=result["x_train"],
        y_train=result["y_train"],
        x_test=result["x_test"],
        y_test=result["y_test"],
    )

    meta = {
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "source_type": result["source_type"],
        "source_detail": result["source_detail"],
        "train_shape": list(result["x_train"].shape),
        "test_shape": list(result["x_test"].shape),
        "train_class_distribution": pd.Series(result["y_train"]).value_counts().sort_index().to_dict(),
        "test_class_distribution": pd.Series(result["y_test"]).value_counts().sort_index().to_dict(),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved preprocessed data: {npz_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Train shape: {result['x_train'].shape}, Test shape: {result['x_test'].shape}")


if __name__ == "__main__":
    main()
