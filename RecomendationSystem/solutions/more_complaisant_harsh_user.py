print("More complaisant user. (only five notations)")
display(rating[rating.userId==rating_gb_user.mean()["rating"].idxmax()][["userId","movie","rating"]])
print("More harsh user. (means of 1.275)")
rating[rating.userId==rating_gb_user.mean()["rating"].idxmin()][["userId","movie","rating"]]