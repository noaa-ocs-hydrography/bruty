import os
from multiprocessing.managers import SyncManager, BaseProxy, AcquirerProxy
from multiprocessing import Lock, Semaphore
import threading
import pathlib
import time
import random

lock_dict = {}
semaphore_dict = {}


class SharableLock:
    def __init__(self, shares=1000):
        self._max_shares = shares
        self.lock = Lock()
        self.semaphore = Semaphore(self._max_shares)
        self._is_locked = False

    def acquire(self, exclusive=True, block=False, timeout=-1, check_interval=1):
        start = time.perf_counter()
        done = False
        while not done:
            if exclusive:
                if not self._is_locked and self.semaphore.get_value() == self._max_shares:
                    ret = self.lock.acquire(block=False)
                    if ret:
                        self._is_locked = True
                        print("lock acquired")
                else:
                    ret = False
            else:
                if not self._is_locked:
                    ret = self.semaphore.acquire(block=False)
                    print('semaphore acquired')
                else:
                    ret = False
            curr = time.perf_counter()
            time_up = timeout > 0 and curr - start > timeout
            done = not block or ret or time_up
            if not done:
                time.sleep(check_interval+random.randrange(0,10)/10.0)
        return ret

    def release(self):
        if self._is_locked:
            print('lock release=', self.lock.release())
            self._is_locked = False
        else:
            print('semaphore release=', self.semaphore.release())

    def locked(self):
        return self._is_locked


# made this in the server since each call to a method opens a new socket because of the multiprocessing.manager
# this class reduces the open socket count greatly
class MultipleLocks:
    def __init__(self, keys=[]):
        self.locks = {}
        self.set_keys(keys)

    def set_keys(self, keys):
        self.release()
        self.keys = [normalize_key(key) for key in keys]

    def get_keys(self):
        return self.keys

    def acquire(self, exclusive=True, block=False, timeout=-1, check_interval=1):
        ret = True
        for key in self.keys:
            lock = lock_dict.setdefault(key, SharableLock())
            if lock.acquire(exclusive=exclusive, block=block, timeout=timeout, check_interval=check_interval):
                self.locks[key] = lock
            else:
                self.release()
                ret = False
                break
        return ret

    def release(self):
        # locks release automatically, but we'll force it rather than wait for garbage collection
        for lock in self.locks.values():
            lock.release()
        self.locks = {}

    def __del__(self):
        self.release()


def normalize_key(key):
    if isinstance(key, pathlib.Path):
        key = str(key)
    if os.name == 'nt':
        if isinstance(key, str):
            key = key.lower()
    return key


def get_lock(key):
    nkey = normalize_key(key)
    # print(f'Lock {nkey} retrieved by thread %s' % (threading.get_ident()))
    return lock_dict.setdefault(nkey, SharableLock())


if __name__ == "__main__":

# def server():
    manager = SyncManager(address=('localhost', 5000), authkey=b'nbslocking')
    manager.register("get_lock", get_lock)  # , AcquirerProxy
    manager.register("MultiLock", MultipleLocks)  # , AcquirerProxy
    server = manager.get_server()
    print(manager)
    print('server running')
    server.serve_forever()

# server()