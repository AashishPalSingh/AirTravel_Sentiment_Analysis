from airTravelSentimentAnalysis import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import string
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from airTravelSentimentAnalysis import logger
from airTravelSentimentAnalysis.entity.config_entity import DataProcessingConfig


class DataProcessing:
    def __init__(self, config: DataProcessingConfig):
        self.config = config

    def loadData(self):
        self.df = pd.read_csv(self.config.raw_data_file, encoding="utf-8")
        return self.df

    def eda(self):
        # Display the first few rows of the DataFrame
        logger.info("\n \n %s", self.df.head())

        logger.info(
            "Unique value count of intent: \n %s",
            self.df[self.config.params_label_col].value_counts(),
        )

        # Display the shape of the DataFrame
        logger.info("Shape of the DataFrame:\n %s", self.df.shape)

        # Display the number of unique values in each column
        logger.info("Unique values: \n %s", self.df.nunique())

        # Display basic statistics for numerical columns
        logger.info("Basic statistics:\n %s", self.df.describe())

        # Display the number of missing values in each column
        logger.info("Missing values:\n %s", self.df.isnull().sum())

        # Display the unique values in the 'intent' column
        logger.info(
            "Unique intents: \n %s", self.df[self.config.params_label_col].unique()
        )

        self.barGraph()
        self.radialGraph()
        return self.df

    def barGraph(self):
        sns.countplot(
            self.df, x=self.config.params_label_col, hue=self.config.params_label_col
        )
        plt.xticks(rotation=90)
        plt.show()

    def radialGraph(self):
        targetCounts = self.df[self.config.params_label_col].value_counts()
        targetLabels = targetCounts.index
        # Make square figures and axes
        plt.figure(1, figsize=(25, 25))
        the_grid = GridSpec(2, 2)
        cmap = plt.get_cmap("coolwarm")
        colors = [cmap(i) for i in np.linspace(0, 1, len(targetLabels))]
        plt.subplot(the_grid[0, 1], aspect=1)

        plt.pie(
            targetCounts,
            labels=targetLabels,
            autopct="%1.1f%%",
            shadow=True,
            colors=colors,
        )
        plt.title("Intent Distribution", fontsize=20)

        plt.show()

    def preProcessData(self):
        # Drop duplicates
        # self.df = self.df.drop_duplicates()

        # Drop null values
        self.df = self.df.dropna()

        self.df[self.config.params_label_col] = (
            self.df[self.config.params_label_col]
            .str.lower()
            .str.strip()
            .replace("_", " ")
        )
        # self.df[config.params_text_col] =  self.df[self.config.params_text_col].apply(self.cleanInstruction)
        return self.df

    def cleanInstruction(instructionText):
        # Remove URLs
        instructionText = re.sub(r"http\S+|www\S+|https\S+", "", instructionText)
        # Remove RT | cc
        instructionText = re.sub(r"RT|cc", "", instructionText)
        # Remove hashtags and mentions
        instructionText = re.sub(r"(@|#)\S+", "", instructionText)
        # Remove punctuations
        instructionText = instructionText.translate(
            str.maketrans("", "", string.punctuation)
        )
        # Remove extra whitespace
        instructionText = re.sub(r"\s+", " ", instructionText).strip()
        return instructionText

    def removeExtraColumns(self):
        # Remove extra columns
        self.df = self.df[[self.config.params_text_col, self.config.params_label_col]]
        return self.df

    def labelEncoding(self):
        # Label Encoding
        le = LabelEncoder()
        copydf = self.df.copy()
        copydf[self.config.params_label_col] = le.fit_transform(
            copydf[self.config.params_label_col]
        )
        self.df = copydf
        return self.df

    def splitData(self):
        # Split the data into train, validation, and test sets
        df = self.df.copy()
        train_test_df, val_df = train_test_split(
            df,
            test_size=self.config.params_test_size,
            stratify=df[self.config.params_label_col],
            random_state=self.config.params_random_state,
        )
        train_df, test_df = train_test_split(
            train_test_df,
            test_size=self.config.params_test_size,
            stratify=train_test_df[self.config.params_label_col],
            random_state=self.config.params_random_state,
        )
        logger.info("Length of train dataframe: \n %s", len(train_df))
        logger.info("Length of test dataframe: \n %s", len(test_df))
        logger.info("Length of val dataframe: \n %s", len(val_df))
        train_df.to_csv(
            self.config.train_data_path, index=False, header=True, encoding="utf-8"
        )
        test_df.to_csv(
            self.config.test_data_path, index=False, header=True, encoding="utf-8"
        )
        val_df.to_csv(
            self.config.val_data_path, index=False, header=True, encoding="utf-8"
        )
