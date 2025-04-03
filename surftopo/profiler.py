import torch
import time

class TimeProfiler:
    """
    Example:
        with Profile("name") as dt:
            pass  # slow operation here
        print(dt)  

        dt = Profile("test_numpy")
        for i in range(10000):
            with dt:
                torch.randn(1000, 1000)        
        ```
    """

    def __init__(self, name=None, cuda_sync=True, registry=None):
        self.name = name
        self.cuda_sync = cuda_sync
        self.registry = registry
        if not torch.cuda.is_available():
            self.cuda_sync = False


    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        end = self.time()
        elapsed = end - self.start
        self.dt = elapsed*1000
        if self.registry is None:
            if self.name is not None:
                print(f"{self.name}: {self}")
        else:
            self.registry.append(self)

    def elapsed_ms(self):
        return self.dt

    def __str__(self):
        return f"{self.dt:.2f} ms"

    def time(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        return time.perf_counter()