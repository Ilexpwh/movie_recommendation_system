# Load the required libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define a function to recommend movies to a user based on similar users
def recommend_movies(user_id):
    # Find the movies that the user hasn't watched yet
    unrated_movies = user_reviews.loc[user_id][user_reviews.loc[user_id] == 0].index
    
    # Find the movies watched by similar users
    similar_users_movies = user_reviews.loc[similar_users[user_id]][unrated_movies]
    
    # Compute the average rating of similar users for each movie
    avg_ratings = similar_users_movies.mean(axis=0)
    
    # Sort the movies by average rating and recommend the top 5
    recommended_movies = avg_ratings.sort_values(ascending=False)[:5].index
    
    return recommended_movies

if __name__ == '__main__':
    # Load the data files
    movie_genres = pd.read_csv('movie_genres.csv', index_col=0)
    user_reviews = pd.read_csv('user_reviews.csv', index_col=0)
    user_reviews['User'] = user_reviews.index

    # Preprocess the data to create a user-item matrix
    user_item_matrix = user_reviews.replace(0, np.nan)

    # Compute the similarity matrix between users
    user_similarity = cosine_similarity(user_item_matrix.fillna(0))

    # Find the top 5 similar users for each user
    similar_users = user_similarity.argsort()[:,::-1][:,:5]
    
    # Call the recommend_movies function for the first 5 users
    for user_id in range(1, 6):
        recommended_movies = recommend_movies(user_id)
        print(f"Recommended movies for User {user_id}:")
        for movie_id in recommended_movies:
            print(f"\t{movie_id}")
