from src.movies_recommendation.exception import CustomException
from src.movies_recommendation.logger import logging
import joblib
import sys
import numpy as np
import pandas as pd
from src.movies_recommendation.exception import CustomException
from sklearn.neighbors import NearestNeighbors

knn_user = joblib.load('E:/moviesrecommendationsystem/artifacts/Knn_user.pkl')
knn_item = joblib.load('E:/moviesrecommendationsystem/artifacts/Knn_item.pkl')
index_data = joblib.load('E:/moviesrecommendationsystem/artifacts/train_target_data_extraction.pkl')
reduced_data = joblib.load('E:/moviesrecommendationsystem/artifacts/train_data_arr_Content_cosine.pkl')



class HybridRecommendationSystem:
    
    def __init__(self):
        try:
            # Load all the necessary components
            self.knn_user = knn_user
            self.knn_item = knn_item
            self.index_data = index_data
            self.reduced_data = reduced_data

        except Exception as e:
            raise CustomException(e, sys)

    def recommend_content_based(self, movie_title):
        try:
            # Find the index of the movie title
            movie_index = self.index_data[self.index_data['title'] == movie_title].index
            if movie_index.empty:
                return [], []

            movie_index = movie_index[0]
            # Use KNN for content-based filtering
            distances_item, indices_item = self.knn_item.kneighbors(self.reduced_data[movie_index:movie_index+1], n_neighbors=6)
            
            # Ensure indices_item[0] is a list of valid indices
            indices = indices_item[0]
            content_based_recommendations = self.index_data.iloc[indices].values.tolist()

            return content_based_recommendations, distances_item[0]

        except Exception as e:
            raise CustomException(e, sys)

    def recommend_collaborative(self, movie_title):
        try:
            movie_index = self.index_data[self.index_data['title'] == movie_title].index
            if movie_index.empty:
                return [], []

            movie_index = movie_index[0]
            # Use KNN for item-based collaborative filtering
            distances_item, indices_item = self.knn_item.kneighbors(self.reduced_data[movie_index:movie_index+1], n_neighbors=6)
            
            # Ensure indices_item[0] is a list of valid indices
            indices = indices_item[0]
            item_based_recommendations = self.index_data.iloc[indices].values.tolist()

            return item_based_recommendations, distances_item[0]
        except Exception as e:
            raise CustomException(e, sys)



    def hybrid_recommendation(self, movie_title):
        try:
            # Get content-based recommendations
            content_based_movies, content_based_scores = self.recommend_content_based(movie_title)

            # Get collaborative recommendations
            collaborative_movies, collaborative_scores = self.recommend_collaborative(movie_title)

            # Combine the recommendations (simple concatenation in this case)
            combined_recommendations = content_based_movies + collaborative_movies
            combined_scores = content_based_scores + collaborative_scores.tolist()

            # Convert lists to tuples to make them hashable for deduplication
            combined_recommendations_tuples = [tuple(rec) if isinstance(rec, list) else rec for rec in combined_recommendations]

            # Remove duplicates using a set
            unique_recommendations = list(set(combined_recommendations_tuples))
            
            # Convert tuples back to lists if needed
            final_recommendations = [list(rec) if isinstance(rec, tuple) else rec for rec in unique_recommendations]

            # Sort scores while keeping track of corresponding recommendations
            final_scores = sorted(zip(final_recommendations, combined_scores), key=lambda x: x[1], reverse=True)
            final_recommendations, final_scores = zip(*final_scores)  # Unzip into two lists

            return list(final_recommendations), list(final_scores)
        

        except Exception as e:
            raise CustomException(e, sys)
