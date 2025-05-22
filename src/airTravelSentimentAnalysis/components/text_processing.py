import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from airTravelSentimentAnalysis.entity.config_entity import TextProcessingConfig


class TextProcessing:
    def __init__(self, config: TextProcessingConfig):
        self.config = config

    def loadData(self):
        self.train_df = pd.read_csv(self.config.train_data_path, encoding="utf-8")
        self.test_df = pd.read_csv(self.config.test_data_path, encoding="utf-8")
        self.val_df = pd.read_csv(self.config.val_data_path, encoding="utf-8")
        return self.train_df, self.test_df, self.val_df

    def createHuggingFaceDataset(self):
        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)
        self.val_dataset = Dataset.from_pandas(self.val_df)
        self.ds = DatasetDict()
        self.ds["train"] = self.train_dataset
        self.ds["test"] = self.test_dataset
        self.ds["val"] = self.val_dataset
        return self.ds

    def tokenize_fn(self, batch):
        tokenizer = AutoTokenizer.from_pretrained(self.config.params_model_name)
        return tokenizer(
            batch[self.config.params_text_col], truncation=True, padding=True
        )

    def tokenizeData(self):
        self.tokenized_datasets = self.ds.map(self.tokenize_fn, batched=True)
        self.tokenized_datasets.save_to_disk(self.config.root_dir)
        return self.tokenized_datasets
