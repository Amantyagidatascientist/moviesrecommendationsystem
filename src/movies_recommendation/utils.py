import os
import pandas as pd
#import pymysql
#from dotenv import load_dotenv
from src.movies_recommendation.exception import CustomException
from src.movies_recommendation.logger import logging
import sys  
import pickle 
import numpy as np
# Import sys for exception handling

#load_dotenv()

# Retrieve environment variables
#host = os.getenv("host")
#user = os.getenv("user")
#password = os.getenv("password")
#db = os.getenv("db")
 #  Make sure to convert the port to an integer
#def read_sql_data():
 #   try:
        # Connect to the MySQL database
  #      mydb = pymysql.connect(
  #          host=host,
  #          user=user,
  #          password=password,
  #          db=db
            # Include port in connection parameters
   #     )
   #     logging.info("Connection established")

        # Execute the query
   #     df = pd.read_sql_query('SELECT * FROM test1', mydb)

    #    return df

    #except Exception as ex:
    #    raise CustomException(ex, sys)
    

def save_object(file_path,obj):
    try :
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:

           pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)