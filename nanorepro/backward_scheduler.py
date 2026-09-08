import threading

class BackwardScheduler:
    def __init__(self, param_buckets, comm_launchers, backward_overlap):
        self.param_buckets = param_buckets
        self.comm_launchers = comm_launchers
        self.backward_overlap = backward_overlap  # if False, then this class becomes no-op
        self._backward_active = False
        self._param_group_bucket_ready = [[False] * len(bucket) for bucket in param_buckets]
        self._next_group_to_launch = 0    # groups need to be launched in exactly same order across ranks to avoid deadlock
        self._backward_lock = threading.Lock()  # param hooks may run from different thread and are not guaranteed to run serially

        if self.backward_overlap:
            for group_idx, bucket in enumerate(param_buckets):
                for param_idx, param in enumerate(bucket):
                    param.register_post_accumulate_grad_hook(self._make_backward_hook(group_idx, param_idx))

    def _make_backward_hook(self, group_idx, param_idx):
        def backward_hook(_):
            if self._backward_active:  # only active in final grad_accum step
                with self._backward_lock:
                    assert not self._param_group_bucket_ready[group_idx][param_idx]
                    self._param_group_bucket_ready[group_idx][param_idx] = True  # mark this param is ready
                    while self._next_group_to_launch < len(self.param_buckets):
                        if all(self._param_group_bucket_ready[self._next_group_to_launch]):  # when all params in group are ready...
                            self.comm_launchers[self._next_group_to_launch]()                # ...launch the reduce-scatter...
                            self._next_group_to_launch += 1                                  # ...and advance to the next group
                        else:
                            break

        return backward_hook

    def backward_overlap_begin(self):
        if not self.backward_overlap:
            return   # no-op if not enabled
        assert not self._backward_active
        self._backward_active = True
        self._next_group_to_launch = 0
        self._param_group_bucket_ready = [[False] * len(bucket) for bucket in self.param_buckets]

    def backward_overlap_end(self):
        if not self.backward_overlap:
            return   # no-op if not enabled
        assert self._backward_active
        assert self._next_group_to_launch == len(self.param_buckets)
        self._backward_active = False
