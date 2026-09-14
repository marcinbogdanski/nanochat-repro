import torch
import threading

class ScheduledBucket:
    def __init__(self, optimizer, bucket_idx, params):
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(bucket_idx, int)
        assert isinstance(params, list) and all(isinstance(p, torch.Tensor) for p in params)

        self.optim = optimizer
        self.bucket_idx = bucket_idx
        self.params = params
        self.ready = [False] * len(params)

    def is_ready(self):
        return all(self.ready)

    def reset(self):
        self.ready = [False] * len(self.params)

class BackwardScheduler:
    def __init__(self, schdeuled_buckets, backward_overlap):
        self._backward_overlap = backward_overlap  # if False, then this class becomes no-op
        if not self._backward_overlap:
            return   # no-op if not enabled

        self._schdeuled_buckets = schdeuled_buckets
        self._param_to_bucket_and_param_idx = {param: (bucket, p_idx) for bucket in schdeuled_buckets for p_idx, param in enumerate(bucket.params)}
        self._backward_active = False
        self._next_bucket_to_launch = 0    # buckets need to be launched in exactly same order across ranks to avoid deadlock
        self._backward_lock = threading.Lock()  # param hooks may run from different thread and are not guaranteed to run serially

        if self._backward_overlap:
            for bucket in self._schdeuled_buckets:
                for param in bucket.params:
                    param.register_post_accumulate_grad_hook(self._param_ready)

    def _param_ready(self, param):
        if self._backward_active:  # only active in final grad_accum step
            with self._backward_lock:
                bucket, param_idx = self._param_to_bucket_and_param_idx[param]
                assert not bucket.ready[param_idx]  # really should not be ready twice
                bucket.ready[param_idx] = True  # mark this param is ready
                while self._next_bucket_to_launch < len(self._schdeuled_buckets):
                    bucket = self._schdeuled_buckets[self._next_bucket_to_launch]
                    if bucket.is_ready():                                  # when all params in group are ready...
                        bucket.optim.launch_reduce(bucket.bucket_idx)      # ...launch the reduce-scatter...
                        self._next_bucket_to_launch += 1                   # ...and advance to the next group
                    else:
                        break

    def backward_overlap_begin(self):
        if not self._backward_overlap:
            return   # no-op if not enabled
        assert not self._backward_active
        self._backward_active = True
        self._next_bucket_to_launch = 0
        for bucket in self._schdeuled_buckets:
            bucket.reset()

    def backward_overlap_end(self):
        if not self._backward_overlap:
            return   # no-op if not enabled
        assert self._backward_active
        assert self._next_bucket_to_launch == len(self._schdeuled_buckets)
        self._backward_active = False
