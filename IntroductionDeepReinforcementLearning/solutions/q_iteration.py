n_steps=10
for step in range(n_steps):
    q_values_ = copy.deepcopy(q_values)
    for state in range(n_states):
        for action in range(n_actions):
            qas = 0
            if transition_probabilities[state][action] is not None:
                for next_state in range(n_states):
                    qas += transition_probabilities[state][action][next_state] * (rewards[state][action][next_state] + gamma * max(q_values_[next_state]))
            q_values[state][action]= qas