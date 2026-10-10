from dataclasses import dataclass
from pathlib import Path

# resolve path properly 
DATA_PATH = Path(__file__).resolve().parents[2] / "src" / "data" / "chestxray_binary_64.npz"


@dataclass
class Config:
    seed: int = 67
    data_path: str = str(DATA_PATH)

    batch_size: int = 128
    learning_rates: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    batch_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)

    max_epochs: int = 20
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    monitor: str = "loss"