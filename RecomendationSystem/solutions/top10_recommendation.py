seen_movie = list(rating_train.rating.values) + list(rating_test.rating.values)
#Run prediction for all movies
prediction = model.predict([np.expand_dims([user_id for _ in range(max_item_id)],axis=1), np.expand_dims([x for x in range(max_item_id)], axis=1)])
#Concatenate results with id of the movie
prediction_with_id = zip(prediction, [x for x in range(max_item_id)])
# Filter on unseen movie, get the title and sort the results according to predicted rate
prediction_of_unseen_movie = sorted([[p[0],id_item_to_title[x]] for p,x in prediction_with_id if not(x in seen_movie)], key=lambda x :x[0], reverse = True)
#Display it.
pd.DataFrame(prediction_of_unseen_movie)