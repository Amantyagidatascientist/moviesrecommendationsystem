from src.movies_recommendation.logger import logging
from src.movies_recommendation.exception import CustomException
import sys
from src.movies_recommendation.utils import read_sql_data
from src.movies_recommendation.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    logging.info("This execution has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()  # Corrected the method name here
    except Exception as e:
        logging.error("Custom Exception occurred", exc_info=True)  # Use logging.error to capture traceback
        raise CustomException(e, sys)
