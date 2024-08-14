from src.movies_recommendation.components import data_tranformation
from src.movies_recommendation.logger import logging
from src.movies_recommendation.exception import CustomException
import sys
from src.movies_recommendation.components.data_ingestion import DataIngestion
from src.movies_recommendation.components.data_tranformation import DataTransformationConfig,DataTransformation

if __name__ == "__main__":
    logging.info("This execution has started")

    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()  # Corrected the method name here

        # Step 2: Data Transformation
        data_transformation = DataTransformation()  # Properly instantiate the class
        data_transformation.initiate_data_transformation(train_path, test_path)  # Fixed typo here

    except Exception as e:
        logging.error("Custom Exception occurred", exc_info=True)  # Use logging.error to capture traceback
        raise CustomException(e, sys)

