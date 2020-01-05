userId=1
# Get list of movies already rated by the user
idmovies_rated_per_user = rating[rating.userId==userId]["movieId"].values
# get prediction fo all movies for movies that are not already rated
predicted = [[mid,UUFilter.predict(userId, mid)] for mid in movies.movieId.values if not(mid in idmovies_rated_per_user)]
# sort predicted list according to the estimation computed
recommendation = sorted(predicted, key=lambda x : x[1].est, reverse=True)
#display the most 10 prediciton with a dataframe
pd.DataFrame([(id_to_title[r[0]], r[1].est) for r in recommendation[:10]])