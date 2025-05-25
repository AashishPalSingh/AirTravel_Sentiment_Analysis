from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import load_from_disk
import numpy as np
import os
from airTravelSentimentAnalysis import logger
from airTravelSentimentAnalysis.entity.config_entity import ModelTrainingConfig
import dagshub
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def compute_metrics(self, eval_pred):
        print(
            "*******************************************************************compute_metrics called +++++++++++++++++++++++++++++++++++++++++++++++"
        )
        try:
            logits, labels = eval_pred
            print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            predictions = np.argmax(
                logits, axis=-1
            )  # Predicted class is the index of max logit
            print(f"Predictions shape: {predictions.shape}")
            print(f"Unique labels: {np.unique(labels)}")
            print(f"Unique predictions: {np.unique(predictions)}")

            precision = precision_score(
                labels, predictions, average="weighted", zero_division=0
            )
            recall = recall_score(
                labels, predictions, average="weighted", zero_division=0
            )
            f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
            accuracy = accuracy_score(labels, predictions)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            print(f"Computed metrics: {metrics}")
            return metrics
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            return {}

    def train(self):
        # dagshub.init(
        #     repo_owner=self.config.params_dagshub_repo_name,
        #     repo_name=self.config.params_dagshub_repo_owner,
        #     mlflow=self.config.params_dagshub_mlflow,
        # )
        # os.environ["MLFLOW_TRACKING_URI"] = (
        #     "https://dagshub.com\/ashish.student2025\/AirTravel_SentimentAnalysis.mlflow"
        # )
        # os.environ["MLFLOW_TRACKING_USERNAME"] = "ashish.student2025"
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = (
        #     "07d3be9cdd70cb406afb2999cb70a58218c12969"
        # )
        # print("*******************************************************************")
        # print(os.environ["MLFLOW_TRACKING_URI"].split(os.sep)[-2])
        # mlflow.set_tracking_uri(
        #     "https://dagshub.com/ashish.student2025/AirTravel_SentimentAnalysis.mlflow"
        # )

        os.environ["MLFLOW_TRACKING_URI"] = self.config.params_mlflow_tracking_uri
        print("*******************************************************************")
        print(os.environ["MLFLOW_TRACKING_URI"].split(os.sep))
        mlflow.set_tracking_uri(self.config.params_mlflow_tracking_uri)
        mlflow.set_experiment(self.config.params_mlflow_experiment_name)
        print("*******************************************************************")
        print(os.environ["MLFLOW_TRACKING_URI"].split(os.sep))
        os.environ["MLFLOW_TRACKING_URI"] = (
            "https://dagshub.com\\ashish.student2025\AirTravel_SentimentAnalysis.mlflow"
        )
        print("*******************************************************************")
        print(os.environ["MLFLOW_TRACKING_URI"].split(os.sep))
        training_args = TrainingArguments(
            output_dir=self.config.model_path,
            evaluation_strategy=self.config.params_eval_strategy,
            save_strategy=self.config.params_save_strategy,
            learning_rate=self.config.params_learning_rate,
            per_device_train_batch_size=self.config.params_per_device_train_batch_size,
            per_device_eval_batch_size=self.config.params_per_device_eval_batch_size,
            num_train_epochs=self.config.params_num_train_epochs,
            weight_decay=self.config.params_weight_decay,
            load_best_model_at_end=self.config.params_load_best_model_at_end,
            metric_for_best_model=self.config.params_metric_for_best_model,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model_path, num_labels=self.config.params_num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_tokenizer_path)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=1,
            lora_alpha=1,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
        )
        peft_model = get_peft_model(model, lora_config)
        tokenized_train_dataset = load_from_disk(self.config.train_tokenized_data_path)
        tokenized_train_dataset = tokenized_train_dataset.rename_column(
            "intent", "labels"
        )
        tokenized_test_dataset = load_from_disk(self.config.test_tokenized_data_path)
        tokenized_test_dataset = tokenized_test_dataset.rename_column(
            "intent", "labels"
        )
        tokenized_val_dataset = load_from_disk(self.config.val_tokenized_data_path)
        tokenized_val_dataset = tokenized_val_dataset.rename_column("intent", "labels")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
        )
        print(
            "*********************** %s ********************************************",
            self.config.params_mlflow_run_name,
        )
        with mlflow.start_run(run_name=self.config.params_mlflow_run_name) as run:
            trainer.train()
            metrics = trainer.evaluate()
            mlflow.log_metrics(metrics)
        trainer.save_model(self.config.model_path)
        tokenizer.save_pretrained(self.config.tokenizer_path)
        tuned_pipeline = pipeline(
            task="text-classification",
            model=trainer.model,
            batch_size=8,
            tokenizer=tokenizer,
            device="cpu",
        )
        model_config = {"batch_size": 8}
        print("Run id %s" % run.info.run_id)
        with mlflow.start_run(run_id=run.info.run_id):
            model_info = mlflow.transformers.log_model(
                transformers_model=tuned_pipeline,
                artifact_path="fine_tuned",
                task="text-classification",
                model_config=model_config,
            )
        print("Model saved in run %s" % model_info.model_uri)
        logger.info("Model saved in run %s", model_info.model_uri)
        mlflow.end_run()
