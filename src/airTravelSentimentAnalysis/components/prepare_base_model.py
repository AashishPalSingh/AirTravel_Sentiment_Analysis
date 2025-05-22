from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from airTravelSentimentAnalysis.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

from airTravelSentimentAnalysis import logger
from torchinfo import summary


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.params_checkpoint, num_labels=self.config.params_num_labels
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
        for name, param in self.model.named_parameters():
            if name.startswith("distilbert"):
                param.requires_grad = False
        logger.info("Base model summary: \n %s", summary(self.model))
        return self.model

    def get_base_model_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.params_checkpoint)

        self.save_model(path=self.config.base_tokenizer_path, model=self.tokenizer)

    @staticmethod
    def save_model(path: Path, model: AutoModelForSequenceClassification):
        model.save_pretrained(path)

    @staticmethod
    def save_tokenizer(path: Path, tokenizer: AutoTokenizer):
        tokenizer.save_pretrained(path)
