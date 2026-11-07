
from pathlib import Path
import pandas as pd


# Location of the extracted ASVspoof 2019 dataset
DATASET_ROOT = Path(
    r"C:\Users\Ishan\Datasets\ASVspoof2019\LA"
)

# Folder names for each dataset split
SPLITS = {
    "train": {
        "audio_folder": "ASVspoof2019_LA_train",
        "protocol_file": "ASVspoof2019.LA.cm.train.trn.txt",
    },
    "dev": {
        "audio_folder": "ASVspoof2019_LA_dev",
        "protocol_file": "ASVspoof2019.LA.cm.dev.trl.txt",
    },
    "eval": {
        "audio_folder": "ASVspoof2019_LA_eval",
        "protocol_file": "ASVspoof2019.LA.cm.eval.trl.txt",
    },
}


def load_protocol(split="train"):
    if split not in SPLITS:
        raise ValueError(
            "split must be 'train', 'dev', or 'eval'"
        )

    split_info = SPLITS[split]

    protocol_path = (
        DATASET_ROOT
        / "ASVspoof2019_LA_cm_protocols"
        / split_info["protocol_file"]
    )

    audio_dir = (
        DATASET_ROOT
        / split_info["audio_folder"]
        / "flac"
    )

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Protocol file not found: {protocol_path}"
        )

    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Audio folder not found: {audio_dir}"
        )

    df = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        header=None,
        names=[
            "speaker_id",
            "file_id",
            "system_id",
            "unused",
            "label",
        ],
    )

    # Convert labels into numeric targets:
    # 0 = bonafide (real)
    # 1 = spoof (fake)
    df["target"] = df["label"].map({
        "bonafide": 0,
        "spoof": 1,
    })

    if df["target"].isna().any():
        raise ValueError(
            "Unexpected labels found in the protocol file."
        )

    df["audio_path"] = df["file_id"].map(
        lambda file_id: str(audio_dir / f"{file_id}.flac")
    )

    missing_count = sum(
        not Path(path).is_file()
        for path in df["audio_path"]
    )

    if missing_count:
        raise FileNotFoundError(
            f"{missing_count} audio files listed in the "
            f"{split} protocol are missing."
        )

    return df


if __name__ == "__main__":
    data = load_protocol("train")

    print("Dataset loaded successfully!")
    print("Number of audio files:", len(data))
    print("\nLabel counts:")
    print(data["label"].value_counts())
    print("\nFirst five records:")
    print(data[
        ["file_id", "label", "target", "audio_path"]
    ].head())