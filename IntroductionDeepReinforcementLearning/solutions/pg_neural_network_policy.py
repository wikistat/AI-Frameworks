def neural_network_policy(observation, model):
    p_right = model.predict(np.expand_dims(observation,axis=0))
    action = 1 if p_right>0.5 else 0
    return action