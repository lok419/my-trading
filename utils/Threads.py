from concurrent.futures import ThreadPoolExecutor

class CustomThreadPool(object):

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomThreadPool, cls).__new__(cls)
            cls._instance.thread_pool = ThreadPoolExecutor(max_workers=1000)
        return cls._instance
    
    def set_max_workers(self, max_workers):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, target, args=[], kwargs={}):
        future = self.thread_pool.submit(target, *args, **kwargs)
        return future

    def submit_many(self, is_multi_thread, target, args=[], kwargs=[]):
        threads = []
        outputs = []

        if max(len(args), len(kwargs)) <= 1:
            is_multi_thread = False

        for i in range(max(len(args), len(kwargs))):
            arg = args[i] if i < len(args) else []
            kwarg = kwargs[i] if i < len(kwargs) else {}

            if is_multi_thread:
                threads.append(self.submit(target, args=arg, kwargs=kwarg))
            else:
                outputs.append(target(*arg, **kwarg))

        if is_multi_thread:
            outputs = [t.result() for t in threads]

        return outputs