import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt


if __name__ == '__main__':   
    # Load user-movie ratings and movie genres data
    user_ratings = pd.read_csv('user_reviews.csv')
    movie_genres = pd.read_csv('movie_genres.csv')
    user_ratings['User'] = user_ratings.index
    # Impute missing ratings with the average rating of the user or movie
    user_ratings = user_ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
    user_ratings = user_ratings.apply(lambda col: col.fillna(col.mean()), axis=0)

    # Split data into training and testing sets
    train_data = user_ratings.sample(frac=0.8, random_state=0)
    test_data = user_ratings.drop(train_data.index)

    # Create user-movie matrix from training data
    user_movie_matrix = train_data.values

    # Perform SVD on the user-movie matrix
    u, s, vt = svds(user_movie_matrix, k=20)
    s_diag_matrix = np.diag(s)
    x_pred = np.dot(np.dot(u, s_diag_matrix), vt)

    # Calculate RMSE on the testing data
    test_data = test_data.values
    test_predictions = x_pred[test_data.nonzero()]
    test_truth = test_data[test_data.nonzero()]
    rmse = sqrt(mean_squared_error(test_predictions, test_truth))
    print("RMSE on the testing data:", rmse)

    # Generate recommendations for each user
    num_recommendations = 5
    user_ids = user_ratings.index
    movie_ids = user_ratings.columns
    
    for user_id in range(10):
        user_idx = np.where(user_ids == user_id)[0][0]
        user_ratings = x_pred[user_idx]
        movies_rated = np.where(user_movie_matrix[user_idx] != 0)[0]
        unrated_movies = np.where(user_movie_matrix[user_idx] == 0)[0]
        sorted_ratings = user_ratings[unrated_movies].argsort()[::-1]
        top_movies = unrated_movies[sorted_ratings][:num_recommendations]
        print("Top {} recommended movies for user {}: {}".format(num_recommendations, user_id, ', '.join([movie_ids[movie_id] for movie_id in top_movies])))
