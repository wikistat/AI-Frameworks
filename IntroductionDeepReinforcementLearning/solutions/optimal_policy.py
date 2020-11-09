def optimal_policy(state, q_values=q_values):
    return np.argmax(q_values[state])