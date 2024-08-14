import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.movies_recommendation.utils import save_object
from src.movies_recommendation.exception import CustomException
from src.movies_recommendation.logger import logging
import os

@dataclass
class DataTransformationConfig:
    predecessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns)
        return X.drop(columns=self.columns)

class DropDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates()

class splitcate(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            X[col] = X[col].str.split()
        return X

class FilterZeroRuntimeBudget(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[~((X['runtime'] == 0) & (X['budget'] == 0))]

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = ['title', 'status', 'original_language', 'original_title', 'overview',
                                   'tagline', 'genres', 'production_companies', 'production_countries', 'spoken_languages',
                                   'keywords']

            Numeric_columns = ['id', 'vote_average', 'vote_count', 'revenue', 'runtime', 'adult', 'popularity', 'budget']

            columns_drop = ['homepage', 'imdb_id', 'backdrop_path', 'poster_path']
            columns_split = ['genres', 'production_companies', 'production_countries', 'keywords']

            sub_pipeline_1 = Pipeline(steps=[
                ('drop_duplicates', DropDuplicates()),
                ('split_cate', splitcate(columns=columns_split)),
                ('filter_zero_runtime_budget', FilterZeroRuntimeBudget()),
                ('drop_columns', DropColumns(columns=columns_drop))
            ])

            categorical_pipeline = Pipeline(steps=[
                ('fill_categorical_values', SimpleImputer(strategy='constant', fill_value='Unknown'))
            ])
            numeric_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])
            release_date_pipeline = Pipeline(steps=[
                ('release_date', SimpleImputer(strategy='most_frequent'))
            ])

            columns_transformer = ColumnTransformer(transformers=[
                ('categorical_columns', categorical_pipeline, categorical_columns),
                ('numeric_pipeline', numeric_pipeline, Numeric_columns),
                ('release_date_pipeline', release_date_pipeline, ['release_date']),
            ], remainder='passthrough')

            sub_pipeline_2 = Pipeline(steps=[
                ('columns_transform_data', columns_transformer)
            ])

            final_pipeline = Pipeline(steps=[
                ('data_cleaning', sub_pipeline_1),
                ('columns_transformation', sub_pipeline_2)
            ])

            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"Numeric columns: {Numeric_columns}")
            logging.info(f"Columns split: {columns_split}")
            logging.info(f"Columns drop: {columns_drop}")

            return final_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read the train data and test data files")
            preprocessing_obj = self.get_data_transformer_object()

            train_data_arr = preprocessing_obj.fit_transform(train_data)
            test_data_arr = preprocessing_obj.transform(test_data)

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_tranformation_config.predecessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing object saved successfully")
        except Exception as e:
            logging.error("Custom Exception occurred", exc_info=True)  # Use logging.error to capture traceback
            raise CustomException(e, sys)

           
