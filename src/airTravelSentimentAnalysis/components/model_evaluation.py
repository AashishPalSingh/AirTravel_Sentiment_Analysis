from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import load_from_disk
from airTravelSentimentAnalysis import logger
from pathlib import Path
import numpy as np
import os
from airTravelSentimentAnalysis.entity.config_entity import ModelTrainingConfig
from airTravelSentimentAnalysis.utils.common import save_json
import dagshub
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel, PeftConfig


class ModelEvaluation:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def getValDataset(self):
        tokenized_val_dataset = load_from_disk(self.config.val_tokenized_data_path)
        tokenized_val_dataset = tokenized_val_dataset.rename_column(
            self.config.params_label_col, "labels"
        )
        return tokenized_val_dataset

    def getBaseModel(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model_path,
            use_safetensors=True,
            trust_remote_code=True,
        )
        return base_model

    def getPeftModel(self):
        peft_config = PeftConfig.from_pretrained(self.config.model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            use_safetensors=True,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, self.config.model_path)
        return peft_model

    def evaluate_base_model(self):
        tokenized_val_dataset = self.getValDataset()
        model = self.getBaseModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Convert dataset to PyTorch format
        tokenized_val_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        dataloader = DataLoader(tokenized_val_dataset, batch_size=8)

        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        logger.info("For base model: \n")
        logger.info("Validation Accuracy: %.4f  \n", accuracy)
        logger.info("Validation F1 Score: %.4f  \n", f1)
        logger.info("Validation Precision: %.4f  \n", precision)
        logger.info("Validation Recall: %.4f  \n", recall)
        scores = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
        self.save_score(scores, "base_model_scores.json")

    def save_score(self, scores, name):
        save_json(path=Path(name), data=scores)

    def evaluatePeftModel(self):
        tokenized_val_dataset = self.getValDataset()
        model = self.getPeftModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Convert dataset to PyTorch format
        tokenized_val_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        dataloader = DataLoader(tokenized_val_dataset, batch_size=8)

        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        logger.info("For Peft model: \n")
        logger.info("Validation Accuracy: %.4f  \n", accuracy)
        logger.info("Validation F1 Score: %.4f  \n", f1)
        logger.info("Validation Precision: %.4f  \n", precision)
        logger.info("Validation Recall: %.4f  \n", recall)
        scores = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
        self.save_score(scores, "peft_model_scores.json")

    def evaluate(self):
        # self.evaluate_base_model()
        self.evaluatePeftModel()
