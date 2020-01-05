print("Most Rated Movies")
display(rating_gb_movie.count()["userId"].sort_values()[::-1][:10])
nb_rate_min = 10
rating_gb_movie_filter = rating_gb_movie.mean()["rating"][rating_gb_movie.count()["userId"]>nb_rate_min]
sorted_movies = rating_gb_movie_filter.sort_values()
print("Top 10 worst movies (for movies with at least 10 rates)")
display(sorted_movies[:10])
print("Top 10 best movies (for movies with at least 10 rates)")
sorted_movies[::-1][:10]