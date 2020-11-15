def simple_policy(observation):
    angle = observation[2]
    action = 0 if angle < 0 else 1
    return action