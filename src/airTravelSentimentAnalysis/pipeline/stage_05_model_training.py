from airTravelSentimentAnalysis.config.configuration import ConfigurationManager
from airTravelSentimentAnalysis.components.model_training import ModelTraining
from airTravelSentimentAnalysis import logger


STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.train()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
