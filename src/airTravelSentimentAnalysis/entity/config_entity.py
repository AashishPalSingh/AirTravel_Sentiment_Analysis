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


@dataclass(frozen=False)
class TextProcessingConfig:
    root_dir: Path
    train_tokenized_data_path: Path
    test_tokenized_data_path: Path
    val_tokenized_data_path: Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    params_model_name: str
    params_text_col: str


@dataclass(frozen=False)
class ModelTrainingConfig:
    root_dir: Path
    base_model_path: Path
    base_tokenizer_path: Path
    model_path: Path
    tokenizer_path: Path
    train_tokenized_data_path: Path
    test_tokenized_data_path: Path
    val_tokenized_data_path: Path
    params_model_name: str
    params_num_labels: int
    params_eval_strategy: str
    params_save_strategy: str
    params_learning_rate: float
    params_per_device_train_batch_size: int
    params_per_device_eval_batch_size: int
    params_num_train_epochs: int
    params_weight_decay: float
    params_load_best_model_at_end: bool
    params_metric_for_best_model: str
    params_dagshub_repo_owner: str
    params_dagshub_repo_name: str
    params_dagshub_mlflow: bool
    params_mlflow_tracking_uri: str
    params_mlflow_experiment_name: str
    params_mlflow_run_name: str
