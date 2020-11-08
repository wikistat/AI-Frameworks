from DQN import DQN
import numpy as np
import collections

# Test 1 Update weight copy weight
dqn = DQN()
for target_layer_weight, main_layer_weight in zip(dqn.target_qn.model.get_weights(), dqn.main_qn.model.get_weights()):
    if len(target_layer_weight.shape)>1:
        assert not(np.all(target_layer_weight == main_layer_weight))

dqn.update_target_graph()
for target_layer_weight, main_layer_weight in zip(dqn.target_qn.model.get_weights(), dqn.main_qn.model.get_weights()):
    if len(target_layer_weight.shape)>1:
        assert np.all(target_layer_weight == main_layer_weight)

#Test 2
dqn = DQN()
state = dqn.env.reset()
# Random action if less than min_pre_train_episode has been played
actions = [dqn.choose_action(state=state,num_episode=99, prob_random=0) for _ in range(100)]
count_action = collections.Counter(actions)
print(count_action)
assert count_action[0]>15
assert count_action[1]>15
assert count_action[2]>15
assert count_action[3]>15

# Random action if we play more than min_pre_train_episode and prob_random is 1
actions = [dqn.choose_action(state=state,num_episode=101, prob_random=1) for _ in range(100)]
count_action = collections.Counter(actions)
print(count_action)
assert count_action[0]>15
assert count_action[1]>15
assert count_action[2]>15
assert count_action[3]>15

# Best action according to model if we play more than min_pre_train_episode and prob_random is 0
actions = [dqn.choose_action(state=state,num_episode=101, prob_random=0) for _ in range(100)]
count_action = collections.Counter(actions)
print(count_action)
assert(len(set(actions)))==1
main_action = list(set(actions))[0]

actions = [dqn.choose_action(state=state,num_episode=101, prob_random=0.5) for _ in range(100)]
count_action = collections.Counter(actions)
assert(len(set(actions)))==4
print(count_action)
assert sorted(count_action.items(), key=lambda x : x[1])[-1][0]==main_action


#Test 3 Run episode generate episode in good format (state, action, reward, next_state, done)
dqn = DQN()
experiences_episode = dqn.run_one_episode(num_episode=200,
                          prob_random=1)

for experience in experiences_episode:
    state, action, reward, next_state, done = experience

    assert state.shape == (84, 84, 3)
    assert type(action) is int
    assert type(reward) is float
    assert next_state.shape == (84, 84, 3)
    assert type(done) is bool
assert len(experiences_episode)<=dqn.max_num_step

# Test 4
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

assert target_q.shape == (2,4)