from airTravelSentimentAnalysis.config.configuration import ConfigurationManager
from airTravelSentimentAnalysis.components.text_processing import TextProcessing
from airTravelSentimentAnalysis import logger


STAGE_NAME = "Text Processing Stage"


class TextProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        text_processing_config = config.get_text_processing_config()
        text_processing = TextProcessing(config=text_processing_config)

        train_df, test_df, val_df = text_processing.loadData()
        print("Load after processed data successfully")
        print(train_df.head())
        logger.info("Load after processed data successfully \n %s", train_df.head())

        ds = text_processing.createHuggingFaceDataset()
        print("Hugging face dataset created successfully")
        print(ds)
        logger.info("Hugging face dataset created successfully \n %s", ds)

        ds = text_processing.createHuggingFaceDataset()
        print("Hugging face dataset created successfully")
        print(ds)
        logger.info("Hugging face dataset created successfully \n %s", ds)

        ds = text_processing.tokenizeData()
        print("Tokenized dataset created successfully")
        print(ds)
        logger.info("Tokenized dataset created successfully \n %s", ds)


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        obj = TextProcessingPipeline()
        obj.main()
        logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
