from DQN_cartpole_solution import DQN
import numpy as np
import collections

# Test 1
dqn = DQN()
dqn.batch_size=2
dqn.save_experience(1)
assert dqn.memory == [1]
dqn.save_experience(2)
assert dqn.memory == [1,2]
dqn.save_experience(3)
assert dqn.memory == [2,3]


#Test 2
dqn = DQN()
state = np.expand_dims(dqn.env.reset(), axis=0)
# Random action if prob random is equal to one
actions = [dqn.choose_action(state=state, prob_random=1) for _ in range(100)]
count_action = collections.Counter(actions)
print(count_action)
assert count_action[0]>35
assert count_action[1]>35
# Best action according to model if prob_random is 0
actions = [dqn.choose_action(state=state, prob_random=0) for _ in range(100)]
count_action = collections.Counter(actions)
print(count_action)
assert(len(set(actions)))==1
main_action = list(set(actions))[0]

actions = [dqn.choose_action(state=state, prob_random=0.5) for _ in range(100)]
count_action = collections.Counter(actions)
assert(len(set(actions)))==2
print(count_action)
assert sorted(count_action.items(), key=lambda x : x[1])[-1][0]==main_action


#Test 3 Run episode generate episode in good format (state, action, reward, next_state, done)
dqn = DQN()
state = np.expand_dims(dqn.env.reset(), axis=0)
state, action, reward, next_state, done  = dqn.run_one_step(state)
assert state.shape == (1, 4)
assert type(action) is int
assert type(reward) is float
assert next_state.shape == (1, 4)
assert type(done) is bool


# Test 3
dqn = DQN()
dqn.batch_size=2
state = np.expand_dims(dqn.env.reset(), axis=0)
target_q = dqn.generate_target_q(
    train_state = np.vstack([state,state]),
    train_action = [0,0],
    train_reward = [1.0,2.0],
    train_next_state = np.vstack([state,state]),
    train_done = [1, 1]
)

assert target_q.shape == (2,2)