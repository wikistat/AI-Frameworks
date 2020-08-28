def get_features_mean(tokens, model):
    features = [model[x] for x in tokens if x in model]
    return np.mean(features, axis=0)