import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, capacity, ttl, clock=time.monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self._d = OrderedDict()   # key -> (value, stored_at, ttl), least recently used first

    def _dead(self, e):
        return self.clock() - e[1] >= e[2]

    def put(self, key, value, ttl=None):
        self._d.pop(key, None)
        if len(self._d) >= self.capacity:
            self.purge()
        while len(self._d) >= self.capacity:
            self._d.popitem(last=False)
        self._d[key] = (value, self.clock(), self.ttl if ttl is None else ttl)

    def get(self, key, default=None):
        e = self._d.get(key)
        if e is None:
            return default
        if self._dead(e):
            del self._d[key]
            return default
        self._d.move_to_end(key)
        return e[0]

    def delete(self, key):
        e = self._d.pop(key, None)
        return e is not None and not self._dead(e)

    def purge(self):
        dead = [k for k, e in self._d.items() if self._dead(e)]
        for k in dead:
            del self._d[k]
        return len(dead)

    def keys(self):
        return [k for k, e in self._d.items() if not self._dead(e)]

    def __len__(self):
        return len(self.keys())
