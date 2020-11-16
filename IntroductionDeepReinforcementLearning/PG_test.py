import numpy as np
from PG_solution import PG


pg = PG()
score = pg.run_one_episode()
assert type(score) is float
for state, action, dreward in pg.experiences:
    assert np.all(state.shape==(1,4))
    assert type(action)==int
    assert type(dreward)==np.float64

loss = pg.run_one_batch_train()
assert type(loss) == float
assert len(pg.experiences)==0