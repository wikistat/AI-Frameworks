def policy_fire(state):
    return [0, 2, 1][state]

all_score = []
for episode in range(1000):
    all_score.append(run_episode(policy_fire, n_steps=100))
print("Summary: mean={:.1f}, std={:1f}, min={}, max={}".format(np.mean(all_score), np.std(all_score), np.min(all_score), np.max(all_score)))


def policy_safe(state):
    return [0, 0, 1][state]

all_score = []
for episode in range(1000):
    all_score.append(run_episode(policy_safe, n_steps=100))
print("Summary: mean={:.1f}, std={:1f}, min={}, max={}".format(np.mean(all_score), np.std(all_score), np.min(all_score), np.max(all_score)))
