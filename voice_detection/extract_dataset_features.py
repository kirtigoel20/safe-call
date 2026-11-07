
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset_loader import load_protocol
from feature_extraction import extract_features


OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_split(split):
    print(f"\nLoading {split} dataset...")
    data = load_protocol(split)

    features = []
    labels = []
    file_ids = []

    print(f"Extracting features from {len(data)} recordings...")

    for row in tqdm(
        data.itertuples(index=False),
        total=len(data),
        desc=f"{split} features"
    ):
        vector = extract_features(row.audio_path)

        features.append(vector)
        labels.append(row.target)
        file_ids.append(row.file_id)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(file_ids)

    output_path = OUTPUT_DIR / f"{split}_features.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        file_ids=ids
    )

    print(f"\n{split.capitalize()} extraction complete!")
    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Bonafide recordings:", int(np.sum(y == 0)))
    print("Spoof recordings:", int(np.sum(y == 1)))
    print("Saved to:", output_path.resolve())


if __name__ == "__main__":
    for split in ("train", "dev"):
        process_split(split)

    print("\nAll feature extraction tasks completed!")