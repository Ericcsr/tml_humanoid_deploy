import numpy as np


class ObsQueue:
    def __init__(self, max_size, stride=4):
        self.max_size = max_size * stride
        self.stride = stride
        self._queue = []

    def push(self, obs):
        if len(self._queue) == 0:
            self._queue = self._queue + [obs.copy()] * self.max_size
        elif len(self._queue) == self.max_size:
            self._queue.pop(0)
            self._queue.append(obs.copy())
    def __getitem__(self, idx):
        return self._queue[idx*self.stride]    

    def __len__(self):
        return len(self._queue) // self.stride

    def get_traj(self):
        return list(reversed(self._queue[::-self.stride]))
    

class HistoryBuffer:
    def __init__(self, history_length, obs_names, flatten=True):
        self.history_length = history_length
        self.flatten = flatten
        self.buffer_dict = {}
        for name in obs_names:
            self.buffer_dict[name] = ObsQueue(history_length,stride=1)

    def add(self, name, obs):
        self.buffer_dict[name].push(obs)

    def get_history(self, name):
        if self.flatten:
            return np.hstack(self.buffer_dict[name].get_traj()).flatten()
        return self.buffer_dict[name].get_traj()