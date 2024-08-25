import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from src.movies_recommendation.utils import save_object
from src.movies_recommendation.exception import CustomException
from src.movies_recommendation.logger import logging
import os
from nltk.stem.porter import PorterStemmer


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
            X = pd.DataFrame(X)
        return X.drop(columns=self.columns)

class DropDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates()

class CombineTextColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        else:
            df = pd.DataFrame(X, columns=self.columns)
            return df.apply(lambda x: ' '.join(x.astype(str)), axis=1)

class DateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')  # Handle invalid parsing
        if X[self.date_column].isnull().all():
            raise ValueError(f"Date column '{self.date_column}' could not be parsed into datetime.")
        X['day'] = X[self.date_column].dt.day.fillna(-1).astype(int)  # Fill NaT with a placeholder value (e.g., -1)
        X['month'] = X[self.date_column].dt.month.fillna(-1).astype(int)
        X['year'] = X[self.date_column].dt.year.fillna(-1).astype(int)
        X = X.drop(columns=[self.date_column])
        return X

class FilterZeroRuntimeBudget(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[~((X['runtime'] == 0) & (X['budget'] == 0))]



class SimpleNLTKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, use_stemming=False):
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming else None

    def preprocess(self, text):

        tokens = text.lower().split()  
        
       
        if self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.preprocess(text) for text in X]

 
 


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, csr_matrix):
            return X.toarray()
        else:
            return X

class ConvertToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(str)

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                'status', 'original_language', 'original_title', 'overview',
                'tagline', 'genres', 'production_companies', 'production_countries', 'spoken_languages',
                'keywords'
            ]

            numeric_columns = [
                'id', 'vote_average', 'vote_count', 'revenue', 'runtime', 'adult', 'popularity', 'budget',
                'year', 'month', 'day'
            ]

            columns_drop = ['homepage', 'imdb_id', 'backdrop_path', 'poster_path']
            columns_split = ['genres', 'production_companies', 'production_countries', 'keywords']

            sub_pipeline_1 = Pipeline(steps=[
                ('drop_duplicates', DropDuplicates()),
                ('filter_zero_runtime_budget', FilterZeroRuntimeBudget()),
                ('drop_columns', DropColumns(columns=columns_drop)),
                ('date_extractor', DateExtractor(date_column='release_date'))
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  
                ('combine_text', CombineTextColumns(columns=categorical_columns)),
                ('drop_columns', DropColumns(columns=categorical_columns)),
                ('nltk_preprocessor', SimpleNLTKPreprocessor(use_stemming=True)),               
                ('vectorizer', CountVectorizer(stop_words='english',max_features=500)),
                ('to_dense', DenseTransformer())
            ])

            numeric_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])

            columns_transformer = ColumnTransformer(transformers=[
                ('categorical', categorical_pipeline, categorical_columns),
                ('numeric', numeric_pipeline, numeric_columns),
            ], remainder='passthrough')

            final_pipeline = Pipeline(steps=[
                ('data_cleaning', sub_pipeline_1),
                ('columns_transformation', columns_transformer)
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numeric columns: {numeric_columns}")
            logging.info(f"Columns to drop: {columns_drop}")

            return final_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read the train data and test data files")

            preprocessing_obj = self.get_data_transformer_object()

            sub_pipeline_1 = Pipeline(steps=[
                ('drop_duplicates', DropDuplicates()),
                ('filter_zero_runtime_budget', FilterZeroRuntimeBudget()),
                
            ])
            train_target_data_extraction=sub_pipeline_1.fit_transform(train_data)

            test_target_data_extraction=sub_pipeline_1.transform(test_data)



            target_columns = 'title'

            input_features_train_df = train_data.drop(columns=target_columns, axis=1)
            target_features_train_df = train_target_data_extraction[target_columns]

            input_features_test_df = test_data.drop(columns=[target_columns], axis=1)
            target_features_test_df = test_target_data_extraction[target_columns]

            logging.info(f"Original train shape: {input_features_train_df.shape}")
            logging.info(f"Original test shape: {input_features_test_df.shape}")

            # Ensure alignment by resetting index
            original_train_indices = input_features_train_df.index
            original_test_indices = input_features_test_df.index

            train_data_arr = preprocessing_obj.fit_transform(input_features_train_df)
            test_data_arr = preprocessing_obj.transform(input_features_test_df)

            # Ensure target columns align with transformed data

            logging.info(f"Shape of train_data_arr: {train_data_arr.shape}")
            logging.info(f"Shape of target_features_train_df: {target_features_train_df.shape}")
            logging.info(f"Shape of test_data_arr: {test_data_arr.shape}")
            logging.info(f"Shape of target_features_test_df: {target_features_test_df.shape}")
            

            

           

            
            train_arr = np.c_[train_data_arr, target_features_train_df]
            test_arr = np.c_[test_data_arr, target_features_test_df]


            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_tranformation_config.predecessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing object saved successfully")
        except Exception as e:
            logging.error("Custom Exception occurred", exc_info=True)  # Use logging.error to capture traceback
            raise CustomException(e, sys)
