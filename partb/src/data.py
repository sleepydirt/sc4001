import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SPLITS = ("train", "val", "test")


def load_arrays(path):
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays


def describe_arrays(arrays):
    "Helper to analyse dataset"
    rows = []
    for key in sorted(arrays):
        values = arrays[key]
        rows.append(
            {
                "array": key,
                "shape": tuple(values.shape),
                "dtype": str(values.dtype),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return rows


def to_tensor(images, mean, std):
    # scale to [0, 1], normalise, then add channel axis
    scaled = torch.from_numpy(images).float() / 255.0
    normalised = (scaled - mean) / std
    return normalised.unsqueeze(1)


def prepare_data(config, batch_size=None):
    arrays = load_arrays(config.data_path)

    # scale based on training set
    train_scaled = torch.from_numpy(arrays["train_images"]).float() / 255.0
    mean = train_scaled.mean()
    std = train_scaled.std()

    loaders = {}
    for split in SPLITS:
        images = to_tensor(arrays[f"{split}_images"], mean, std)
        labels = torch.from_numpy(arrays[f"{split}_labels"]).float().unsqueeze(1)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            TensorDataset(images, labels),
            batch_size=batch_size or config.batch_size,
            shuffle=shuffle,
        )

    stats = {
        "mean": mean.item(),
        "std": std.item(),
        "image_shape": tuple(arrays["train_images"].shape[1:]),
        "channels": 1,
        "positive_rate": {
            split: float(arrays[f"{split}_labels"].mean()) for split in SPLITS
        },
    }
    return loaders, stats
