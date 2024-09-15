from src.movies_recommendation.components import data_tranformation
from src.movies_recommendation.logger import logging
from src.movies_recommendation.exception import CustomException
import sys
from src.movies_recommendation.components.data_ingestion import DataIngestion
from src.movies_recommendation.components.data_tranformation import DataTransformationConfig,DataTransformation
from src.movies_recommendation.components.model_trainer import  ModelTrainer
from src.movies_recommendation.pipelines.prediction_pipeline import HybridRecommendationSystem

if __name__ == "__main__":
    logging.info("This execution has started")

    try:
        hybrid_model=HybridRecommendationSystem()


        new=hybrid_model.recommend_collaborative("Iron Man")
        print(new)
        #data_ingestion = DataIngestion()
        #train_path= data_ingestion.initiate_data_ingestion()  # Corrected the method name here

        # Step 2: Data Transformation
        #data_transformation = DataTransformation()  # Properly instantiate the class
        #train_data_arr_Content,train_data_arr_Collaborative,_,_,_,_=data_transformation.initiate_data_transformation(train_path)  # Fixed typo here

        #Model_Trainer=ModelTrainer()
        #Model_Trainer.initiate_model_trainer(train_data_arr_Content,train_data_arr_Collaborative)




    except Exception as e:
        logging.error("Custom Exception occurred", exc_info=True)  # Use logging.error to capture traceback
        raise CustomException(e, sys)

