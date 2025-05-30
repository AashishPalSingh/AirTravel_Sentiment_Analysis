from airTravelSentimentAnalysis.config.configuration import ConfigurationManager
from airTravelSentimentAnalysis.components.model_evaluation import ModelEvaluation
from airTravelSentimentAnalysis import logger


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
