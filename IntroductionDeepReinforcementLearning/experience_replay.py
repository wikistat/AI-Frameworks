import numpy as np
class ExperienceReplay:
    def __init__(self, buffer_size=50000):
        """ Data structure used to hold game experiences """
        # Buffer will contain [state,action,reward,next_state,done]
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experiences):
        """ Adds list of experiences to the buffer """
        # Extend the stored experiences
        self.buffer.extend(experiences)
        # Keep the last buffer_size number of experiences
        self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, size):
        """ Returns a sample of experiences from the buffer """
        sample_idxs = np.random.randint(len(self.buffer), size=size)
        sample_output = [self.buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output, (size, -1))
        return sample_output