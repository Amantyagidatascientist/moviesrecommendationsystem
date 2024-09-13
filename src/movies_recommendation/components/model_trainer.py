import sys
import time
import pandas as pd
from dataclasses import dataclass
from src.movies_recommendation.exception import CustomException
from src.movies_recommendation.logger import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from src.movies_recommendation.utils import save_object
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path1 = os.path.join("artifacts", "knn_user.pkl")
    trained_model_file_path2 = os.path.join("artifacts", "knn_item.pkl")
    trained_model_file_path3 = os.path.join("artifacts", "content_based_similarity.pkl")
    trained_model_file_path4 = os.path.join("artifacts", "train_data_arr_Content_cosine.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def handle_infinity_and_nans(self, array, array_name):
        """Replace inf and NaNs with a finite value or zero."""
        try:
            if isinstance(array, pd.DataFrame):
                array = array.to_numpy()
            array[np.isinf(array)] = 0
            array[np.isnan(array)] = 0
            return array
        except Exception as e:
            logging.error(f"Error handling {array_name} for infinity and NaN values: {str(e)}")
            raise CustomException(e, sys)

    def check_and_convert_to_numeric(self, array, array_name):
        """Ensure that the input data is numeric, and convert to float32 for better precision."""
        try:
            array = self.handle_infinity_and_nans(array, array_name)
            logging.warning(f"Converting {array_name} to float32...")
            array = array.astype(np.float32)
            return array
        except Exception as e:
            logging.error(f"Error converting {array_name} to numeric data: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_data_arr_Content, train_data_arr_Collaborative):
        try:
            logging.info("Starting model training...")

            # Convert arrays to float32 for memory optimization
            train_data_arr_Content = self.check_and_convert_to_numeric(train_data_arr_Content, "train_data_arr_Content")
            train_data_arr_Collaborative = self.check_and_convert_to_numeric(train_data_arr_Collaborative, "train_data_arr_Collaborative")

            # Fit KNN models for both user and item collaborative filtering
            logging.info("Fitting KNN models...")
            start_time = time.time()

            knn_user = NearestNeighbors(n_neighbors=5, metric='cosine')
            knn_user.fit(train_data_arr_Collaborative)

            knn_item = NearestNeighbors(n_neighbors=5, metric='cosine')
            knn_item.fit(train_data_arr_Content)

            logging.info(f"KNN models fitted in {time.time() - start_time:.2f} seconds.")

            # Save the trained models and the content-based data matrix
            logging.info("Saving models and content-based matrix...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path2,  # Save knn_item
                obj=knn_item
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path1,  # Save knn_user
                obj=knn_user
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path4,  # Save train_data_arr_Content
                obj=train_data_arr_Content
            )

            logging.info("Models and matrix saved successfully.")

            return (
                knn_user,
                knn_item,
                train_data_arr_Content
            )
        except Exception as e:
            logging.error("An error occurred during model training.")
            raise CustomException(e, sys)







