import random

class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, discrete_action, continuous_action, reward, next_state, done = zip(*batch)
        return state, discrete_action, continuous_action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


