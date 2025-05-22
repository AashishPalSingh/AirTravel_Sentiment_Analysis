from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=False)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    base_tokenizer_path: Path
    updated_base_tokenizer_path: Path
    params_checkpoint: str
    params_num_labels: int


@dataclass(frozen=False)
class DataProcessingConfig:
    root_dir: Path
    raw_data_file: str
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    params_test_size: float
    params_label_col: str
    params_text_col: str
    params_random_state: int
