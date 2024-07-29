import os
import sys
from src.movies_recommendation.logger import logging
import pandas as pd
from src.movies_recommendation.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.movies_recommendation.utils import read_sql_data

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):  # Corrected the typo here
        try:
            logging.info("Reading from MySQL database")
            df = read_sql_data()
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data Ingestion is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()  # Corrected the typo here
