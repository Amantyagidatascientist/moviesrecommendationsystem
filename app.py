from src.movies_recommendation.logger import logging
from src.movies_recommendation.exception import CustomException
import sys
if __name__=="__main__":
    logging.info("this excecution has started")

    try:
        a=1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)