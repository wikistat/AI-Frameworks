import math
def discount_rewards(rewards, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
    """
    n_rewards = len(rewards)
    discounted_rewards = [0 for _ in range(n_rewards)]
    for i in range(n_rewards):
        for j,r in enumerate(rewards[i:]):
            discounted_rewards[i]+= math.pow(gamma,j)*r
    return discounted_rewards