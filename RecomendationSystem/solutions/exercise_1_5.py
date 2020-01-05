#Load the data
DATA_DIR = "ml-25/"
rating = pd.read_csv(DATA_DIR + "ratings.csv")
nb_entries = rating.shape[0]
print("Number of entries : %d " %nb_entries)

#Load the movies files in order to create the id_to_title dictionary and complete the rating dataframe
movies = pd.read_csv(DATA_DIR + "movies.csv")
id_to_title = dict(movies[["movieId","title"]].values)
rating["movie"] = [id_to_title[x] for x in rating["movieId"].values]
rating.head()

# Create the train/test daaset and save the test ids
rating["test_train"] = ["test" if random.random()<=0.1 else "train" for _ in range(rating.shape[0])]
rating["test_train"].value_counts()
rating.to_csv("ml-25/ratings_updated.csv",index=False)
reader = surprise.Reader(rating_scale=(0, 5))
rating_train = rating[rating.test_train=="train"]
data = surprise.Dataset.load_from_df(rating_train[['userId', 'movieId', 'rating']], reader)
train = data.build_full_trainset()
rating_test = rating[rating.test_train=="test"]
test = list([tuple(x) for x in rating_test[['userId', 'movieId', 'rating']].values])

#Fit and test the algorithm
IIFilter = spa.knns.KNNWithMeans(k=100, 
                      min_k =1, 
                      sim_options = {'name': 'msd',
                                     'user_based': False},
                     verbose=1)
# Train the algorithm on the trainset, and predict ratings for the testset
IIFilter.fit(train)
predictions = UUFilter.test(test)

# Then compute RMSE
surprise.accuracy.rmse(predictions)