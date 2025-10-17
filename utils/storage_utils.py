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