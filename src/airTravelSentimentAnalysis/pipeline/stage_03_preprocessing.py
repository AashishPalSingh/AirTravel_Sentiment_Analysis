from airTravelSentimentAnalysis.config.configuration import ConfigurationManager
from airTravelSentimentAnalysis.components.data_processing import DataProcessing
from airTravelSentimentAnalysis import logger


STAGE_NAME = "Pre Processing Stage"


class PrepareDataProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_processing_config = config.get_data_processing_config()
        data_processing = DataProcessing(config=data_processing_config)
        df = data_processing.loadData()
        print("Data loaded successfully")
        print(df.describe())
        logger.info("Data loaded successfully \n %s", df.describe())
        df = data_processing.eda()
        print("Data EDA processed successfully")
        print(df.describe())
        logger.info("Data EDA processed successfully \n %s", df.describe())
        df = data_processing.preProcessData()
        print("Data preprocessed successfully")
        print(df.describe())
        logger.info("Data preprocessed successfully \n %s", df.describe())
        df = data_processing.labelEncoding()
        print("Label encoded successfully")
        print(df.head())
        logger.info("Label encoded successfully \n %s", df.head())
        df = data_processing.removeExtraColumns()
        print("Removed Extra columns successfully")
        print(df.head())
        logger.info("Removed Extra columns successfully \n %s", df.head())
        data_processing.splitData()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        obj = PrepareDataProcessingPipeline()
        obj.main()
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
