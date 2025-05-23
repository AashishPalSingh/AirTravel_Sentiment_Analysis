from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
import numpy as np
from airTravelSentimentAnalysis.entity.config_entity import ModelTrainingConfig

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(
            logits, axis=-1
        )  # Predicted class is the index of max logit
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self):
        print(self.config)
        training_args = TrainingArguments(
            output_dir=self.config.model_path,
            eval_strategy="epoch",
            save_strategy=self.config.params_save_strategy,
            # learning_rate=self.config.params_learning_rate,
            per_device_train_batch_size=self.config.params_per_device_train_batch_size,
            per_device_eval_batch_size=self.config.params_per_device_eval_batch_size,
            num_train_epochs=self.config.params_num_train_epochs,
            weight_decay=self.config.params_weight_decay,
            load_best_model_at_end=self.config.params_load_best_model_at_end,
            metric_for_best_model=self.config.params_metric_for_best_model,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_tokenizer_path)
        tokenized_train_dataset = load_from_disk(self.config.train_tokenized_data_path)
        tokenized_train_dataset = tokenized_train_dataset.rename_column(
            "intent", "labels"
        )
        tokenized_test_dataset = load_from_disk(self.config.test_tokenized_data_path)
        tokenized_val_dataset = load_from_disk(self.config.val_tokenized_data_path)
        tokenized_val_dataset = tokenized_val_dataset.rename_column("intent", "labels")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        trainer.save_model(self.config.model_path)
        tokenizer.save_pretrained(self.config.tokenizer_path)
