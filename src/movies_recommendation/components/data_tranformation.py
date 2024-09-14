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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix



import os
from nltk.stem.porter import PorterStemmer


@dataclass
class DataTransformationConfig:
    predecessor_obj_file_path1 = os.path.join('artifacts', 'train_data_arr_Content.pkl')
    predecessor_obj_file_path2= os.path.join('artifacts', 'train_target_data_extraction.pkl')
    predecessor_obj_file_path3= os.path.join('artifacts', 'train_data_arr_Collaborative.pkl')

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return X.drop(columns=self.columns)

class RemoveSpace(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Make a copy of X to avoid modifying the original DataFrame

        for column in self.columns:
            # Convert all values to strings and handle NaN values
            X[column] = X[column].astype(str).fillna('')

            # Remove spaces and split/join the string elements
            X[column] = X[column].str.replace(' ', '', regex=False).apply(lambda x: ' '.join(x.split(',')))

        return X
    

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
        X['day'] = X[self.date_column].dt.day.fillna(1).astype(int)  # Fill NaT with a placeholder value (e.g., -1)
        X['month'] = X[self.date_column].dt.month.fillna(1).astype(int)
        X['year'] = X[self.date_column].dt.year.fillna(2000).astype(int)
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

class UserIDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_id=1):
        self.start_id = start_id

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['user_id'] = range(self.start_id, self.start_id + len(X))
        return X

class DropNullProductionData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Drop rows where both 'production_countries' and 'production_companies' are null
        return X[~(X['production_countries'].isnull() & X['production_companies'].isnull())]
    


class DropNonReleasedMovies(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Drop rows where 'status' is not 'Released'
        return X[X['status'] == 'Released']
    

class MovieRatingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.C = None
        self.m = None

    def weighted_rating(self, x):
        v = x['vote_count']
        R = x['vote_average']
        weighted_score = (v / ((v + self.m) * R +0.001)) + (self.m / ((v + v) * self.C +0.001))
        return round(weighted_score, 2)

    def apply_weighted_rating(self, df):
        df['weighted_rating'] = df.apply(lambda x: self.weighted_rating(x), axis=1)
        return df

    def fit(self, X, y=None):
        self.C = X['vote_average'].mean()
        self.m = X['vote_count'].quantile(0.90)
        return self

    def transform(self, X, y=None):
        return self.apply_weighted_rating(X)
    

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

            space_repalce=['production_companies','production_countries','spoken_languages']

            columns_drop = ['homepage', 'imdb_id', 'backdrop_path', 'poster_path']
            columns_split = ['genres', 'production_companies', 'production_countries', 'keywords']

            sub_pipeline_1 = Pipeline(steps=[
                ('drop_duplicates', DropDuplicates()),
                ('filter_zero_runtime_budget', FilterZeroRuntimeBudget()),
                ('DropNullProductionData',DropNullProductionData()),
                ('DropNonReleasedMovies',DropNonReleasedMovies()),
                ('drop_columns', DropColumns(columns=columns_drop)),
                ('date_extractor', DateExtractor(date_column='release_date')),
                ('RemoveSpace',RemoveSpace(columns=space_repalce))
                
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  
                ('combine_text', CombineTextColumns(columns=categorical_columns)),
                ('drop_columns', DropColumns(columns=categorical_columns)),
                ('nltk_preprocessor', SimpleNLTKPreprocessor(use_stemming=True)),               
                ('TfidfVectorizer',TfidfVectorizer(stop_words='english',max_features=5000)),
                ('TruncatedSVD', TruncatedSVD(n_components=1000))
                
                
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
        

    def get_data_transformer_Collaborative(self):
            columns_drop_initial = ['status', 'release_date', 'revenue', 'runtime', 'adult', 
                            'budget', 'homepage', 'imdb_id', 'original_language', 
                            'original_title', 'overview', 'popularity', 'tagline', 
                            'genres', 'production_companies', 'production_countries', 
                            'spoken_languages', 'keywords', 'backdrop_path', 'poster_path']
    
            columns_to_drop_after = ['vote_average', 'vote_count']
    
            sub_pipeline_1 = Pipeline(steps=[
                ('drop_duplicates', DropDuplicates()),  # Assuming these custom transformers exist
                ('filter_zero_runtime_budget', FilterZeroRuntimeBudget()),
                ('DropNullProductionData',DropNullProductionData()),
                ('DropNonReleasedMovies',DropNonReleasedMovies()),
                ('drop_columns', DropColumns(columns=columns_drop_initial)),
                
                ])
    
            final_pipeline = Pipeline(steps=[
                ('sub_pipeline_1', sub_pipeline_1),
                ('movie_rating_transformer', MovieRatingTransformer()),
                ('drop_vote_columns', DropColumns(columns=columns_to_drop_after)),  # Drops 'vote_average' and 'vote_count'
                ('user_id_transformer', UserIDTransformer())
                ])
    
            return final_pipeline
    
    def get_data_test_object(self):

        try:
            categorical_columns=['id',  'vote_average', 'vote_count', 'status', 'release_date',
       'revenue', 'runtime', 'adult', 'budget', 'homepage', 'imdb_id',
       'original_language', 'original_title', 'overview', 'popularity',
       'tagline', 'genres', 'production_companies', 'production_countries',
       'spoken_languages', 'keywords', 'backdrop_path', 'poster_path']
            

            sub_pipeline_1 = Pipeline(steps=[
                ('drop_duplicates', DropDuplicates()),
                ('filter_zero_runtime_budget', FilterZeroRuntimeBudget()),
                ('DropNullProductionData',DropNullProductionData()),
                ('DropNonReleasedMovies',DropNonReleasedMovies())
                
            ])
            

            categorical_pipeline = Pipeline(steps=[
                
                ('drop_columns', DropColumns(columns=categorical_columns)),
                
                
            ])
            

            final_pipeline = Pipeline(steps=[
                ('data_cleaning', sub_pipeline_1),
                ('columns_transformation', categorical_pipeline)
            ])

            return final_pipeline

        except  Exception as e:
            raise CustomException(e, sys)

    

    
    def initiate_data_transformation(self, train_path):
        try:
            train_data = pd.read_csv(train_path)
            

            logging.info("Read the train data and test data files")

            preprocessing_obj = self.get_data_transformer_object()
            preprocessing_obj2=self.get_data_transformer_Collaborative()
            terget_data=self.get_data_test_object()


            
            train_target_data_extraction=terget_data.fit_transform(train_data)

            

            target_columns = 'title'

            input_features_train_df = train_data.drop(columns=target_columns, axis=1)
            target_features_train_df = train_target_data_extraction

            

            logging.info(f"Original train shape: {input_features_train_df.shape}")
                      

            train_data_arr_Content = preprocessing_obj.fit_transform(input_features_train_df)
            train_data_arr_Collaborative=preprocessing_obj2.fit_transform(input_features_train_df)
            train_data_arr_Content=train_data_arr_Content
            train_data_arr_Collaborative=train_data_arr_Collaborative
            train_data_arr_Content_sparse = csr_matrix(train_data_arr_Content)
            target_features_train_df=target_features_train_df

            # Ensure target columns align with transformed data

            logging.info(f"Shape of train_data_arr: {train_data_arr_Content.shape}")
            logging.info(f"Shape of train_data_arr: {train_data_arr_Collaborative.shape}")

            logging.info(f"Shape of target_features_train_df: {target_features_train_df.shape}")
           

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_tranformation_config.predecessor_obj_file_path1,
                obj=train_data_arr_Content_sparse
            )
            save_object(
                file_path=self.data_tranformation_config.predecessor_obj_file_path2,
                obj=train_target_data_extraction
            )
            save_object(
                file_path=self.data_tranformation_config.predecessor_obj_file_path3,
                obj=train_data_arr_Collaborative
            )
            logging.info("Preprocessing object saved successfully")

            return (train_data_arr_Content,
                    train_data_arr_Collaborative,
                    target_features_train_df,
                    self.data_tranformation_config.predecessor_obj_file_path1,
                    self.data_tranformation_config.predecessor_obj_file_path2,
                    self.data_tranformation_config.predecessor_obj_file_path3
                    )
        
        except Exception as e:
            logging.error("Custom Exception occurred", exc_info=True)  # Use logging.error to capture traceback
            raise CustomException(e, sys)