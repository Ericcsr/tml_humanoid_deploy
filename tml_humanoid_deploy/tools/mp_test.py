import time
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory


shm_buffer = []

def shared_np(size, name, dtype=np.float32):
    try:
        shm = SharedMemory(create=True, size=np.prod(size) * np.dtype(dtype).itemsize, name=name)
        arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
        arr[:] = 0.0
        shm_buffer.append(shm) # prevent crash
    except FileExistsError:
        print("Shared memory already exists")
        shm = SharedMemory(create=False, name=name)
        arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
        arr[:] = 0.0
        shm_buffer.append(shm) # prevent crash
    return arr

class SimProcess:
    def __init__(self):
        self.data = shared_np((100,), "test", np.float32)
        self.lock = mp.Lock()
        self.process = mp.Process(target=SimProcess.run, args=(self.lock,))
        self.process.start()

    @staticmethod
    def run(lock):
        # Simulate some work
        print("Process start")
        data = shared_np((100,), "test", np.float32)
        t = 0.0
        while True:
            lock.acquire()
            for i in range(100):
                data[i] = np.sin(t+0.01*i)
            lock.release()
            t += 0.01
            time.sleep(0.01)

    def get_value(self):
        self.lock.acquire()
        value = self.data.copy()
        self.lock.release()
        return value

if __name__ == "__main__":
    sim = SimProcess()
    while True:
        print(sim.get_value())
        time.sleep(0.1)