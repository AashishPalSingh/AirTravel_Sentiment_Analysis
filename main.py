from airTravelSentimentAnalysis import logger
from airTravelSentimentAnalysis.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from airTravelSentimentAnalysis.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from airTravelSentimentAnalysis.pipeline.stage_03_preprocessing import (
    PrepareDataProcessingPipeline,
)

from airTravelSentimentAnalysis.pipeline.stage_04_text_processing import (
    TextProcessingPipeline,
)

from airTravelSentimentAnalysis.pipeline.stage_05_model_training import (
    ModelTrainingPipeline,
)

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model"
try:
    logger.info("*******************")
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Preprocessing stage"
try:
    logger.info("*******************")
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    preprocessing = PrepareDataProcessingPipeline()
    preprocessing.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Text Processing stage"
try:
    logger.info("*******************")
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    textProcessing = TextProcessingPipeline()
    textProcessing.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training stage"
try:
    logger.info("*******************")
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    modelTraining = ModelTrainingPipeline()
    modelTraining.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e
