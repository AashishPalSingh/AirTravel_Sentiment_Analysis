from airTravelSentimentAnalysis.config.configuration import ConfigurationManager
from airTravelSentimentAnalysis.components.prepare_base_model import PrepareBaseModel
from airTravelSentimentAnalysis import logger

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.get_base_model_tokenizer()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
