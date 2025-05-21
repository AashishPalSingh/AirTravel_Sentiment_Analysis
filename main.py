from airTravelSentimentAnalysis import logger
from airTravelSentimentAnalysis.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
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
