from airTravelSentimentAnalysis.constants import *
from airTravelSentimentAnalysis.utils.common import read_yaml, create_directories
from airTravelSentimentAnalysis.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    DataProcessingConfig,
    TextProcessingConfig,
    ModelTrainingConfig,
)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            base_tokenizer_path=Path(config.base_tokenizer_path),
            updated_base_tokenizer_path=Path(config.updated_base_tokenizer_path),
            params_checkpoint=self.params.CHECKPOINT,
            params_num_labels=self.params.NUM_LABELS,
        )
        return prepare_base_model_config

    def get_data_processing_config(self) -> DataProcessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])
        data_processing_config = DataProcessingConfig(
            root_dir=Path(config.root_dir),
            raw_data_file=config.raw_data_file,
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            val_data_path=Path(config.val_data_path),
            params_test_size=self.params.TEST_SIZE,
            params_label_col=self.params.LABEL_COL,
            params_text_col=self.params.TEXT_COL,
            params_random_state=self.params.RANDOM_STATE,
        )
        return data_processing_config

    def get_text_processing_config(self) -> TextProcessingConfig:
        config = self.config.text_processing
        config["train_data_path"] = self.config.data_preprocessing.train_data_path
        config["test_data_path"] = self.config.data_preprocessing.test_data_path
        config["val_data_path"] = self.config.data_preprocessing.val_data_path

        create_directories([config.root_dir])

        text_processing_config = TextProcessingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            val_data_path=Path(config.val_data_path),
            train_tokenized_data_path=Path(config.train_tokenized_data_path),
            test_tokenized_data_path=Path(config.test_tokenized_data_path),
            val_tokenized_data_path=Path(config.val_tokenized_data_path),
            params_model_name=self.params.MODEL_NAME,
            params_text_col=self.params.TEXT_COL,
        )

        return text_processing_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        config["base_model_path"] = self.config.prepare_base_model.base_model_path
        config["base_tokenizer_path"] = (
            self.config.prepare_base_model.base_tokenizer_path
        )
        params_training = self.params.TRAINING_ARGUMENTS
        params_dagshub = self.params.DAGSHUB_ARGUMENTS
        params_mlflow = self.params.MLFLOW_ARGUMENTS
        # print(f"Training arguments: {params_training}")
        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            base_tokenizer_path=Path(config.base_tokenizer_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            train_tokenized_data_path=Path(config.train_tokenized_data_path),
            test_tokenized_data_path=Path(config.test_tokenized_data_path),
            val_tokenized_data_path=Path(config.val_tokenized_data_path),
            params_model_name=self.params.MODEL_NAME,
            params_num_labels=self.params.NUM_LABELS,
            params_eval_strategy=params_training.eval_strategy,
            params_save_strategy=params_training.save_strategy,
            params_learning_rate=float(params_training.learning_rate),
            params_per_device_train_batch_size=params_training.per_device_train_batch_size,
            params_per_device_eval_batch_size=params_training.per_device_eval_batch_size,
            params_num_train_epochs=params_training.num_train_epochs,
            params_weight_decay=params_training.weight_decay,
            params_load_best_model_at_end=params_training.load_best_model_at_end,
            params_metric_for_best_model=params_training.metric_for_best_model,
            params_dagshub_repo_owner=params_dagshub.repo_owner,
            params_dagshub_repo_name=params_dagshub.repo_name,
            params_dagshub_mlflow=params_dagshub.mlflow,
            params_mlflow_tracking_uri=params_mlflow.tracking_uri,
            params_mlflow_experiment_name=params_mlflow.experiment_name,
            params_mlflow_run_name=params_mlflow.run_name,
        )

        return model_training_config
