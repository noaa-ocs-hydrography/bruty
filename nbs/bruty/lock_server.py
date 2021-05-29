import os
import sys
from multiprocessing.managers import SyncManager, BaseProxy, AcquirerProxy
from multiprocessing import Lock, Semaphore
import threading
import pathlib
import time
import random

lock_dict = {}
semaphore_dict = {}


class SharableLock:
    """ Basic class to keep an exclusive lock OR a shared lock (semaphore).
    Wrapped around multiprocessing Lock and Semaphore using the SyncManager to pass back to the callers.
    """
    def __init__(self, name: str, shares: int=1000):
        """ Create a lock which can be used later for exclusive access or sharing up to "shares" times.
        
        Parameters
        ----------
        name : str
            informational name to 
        shares : int
            number of concurrent shares to allow access
        """
        self.name = name
        self._max_shares = shares
        self.lock = Lock()
        self.semaphore = Semaphore(self._max_shares)
        self._is_locked = False

    def acquire(self, exclusive: bool=True, block: bool=False, timeout: float=-1, check_interval: float=1, silent=False):
        """ Acquire the lock, before calling this the instance has no exclusive access or shared access.
        Parameters
        ----------
        exclusive : bool
            If True then the lock will require exclusive access (read/write).  
            If False the a read only shared access is requested.
        block : bool
            If True then wait for the desired access to become available  or the timeout expires.
            If False then return immediately after the first attempt to lock.            
        timeout : float
            Time in seconds to wait before returning if the lock is already in use
        check_interval : float
            Time in seconds to wait between attempts to lock.  Zero may result in deadlocks/

        Returns
        -------

        """
        start = time.perf_counter()
        done = False
        while not done:
            if exclusive:
                if not self._is_locked and self.semaphore.get_value() == self._max_shares:
                    ret = self.lock.acquire(block=False)
                    if ret:
                        self._is_locked = True
                        if not silent:
                            print("lock acquired", self.name)
                else:
                    ret = False
            else:
                if not self._is_locked:
                    ret = self.semaphore.acquire(block=False)
                    if not silent:
                        print('semaphore acquired', self.name)
                else:
                    ret = False
            curr = time.perf_counter()
            time_up = timeout > 0 and curr - start > timeout
            done = not block or ret or time_up
            if not done:
                time.sleep(check_interval+random.randrange(0,10)/10.0)
        return ret

    def release(self, silent=False):
        """  Release the lock if any.
        Returns
        -------
        None

        """
        if self._is_locked:
            self.lock.release()
            if not silent:
                print('lock release', self.name)
            self._is_locked = False
        else:
            self.semaphore.release()
            if not silent:
                print('semaphore release=', self.name)

    def locked(self):
        return self._is_locked


# made this in the server since each call to a method opens a new socket because of the multiprocessing.manager
# this class reduces the open socket count greatly
class MultipleLocks:
    """ Class to control a group of locks at one time.  ys will be treate as one object so will all succeed or all fail.
    """
    def __init__(self, keys=[]):
        self.keys = []
        self.locks = {}
        self.set_keys(keys)

    def set_keys(self, keys):
        """
        """
        self.release()
        self.keys = [normalize_key(key) for key in keys]

    def get_keys(self):
        return self.keys

    def acquire(self, exclusive=True, block=False, timeout=-1, check_interval=1):
        """ This function acquires a lock for each key that had been set by set_keys.
        If any key can not be acquired all locks are released and fails.

        Parameters
        ----------
        exclusive
        block
        timeout
        check_interval

        Returns
        -------
        success : bool
            True if all key locks were acquired

        """
        ret = True
        print("acquiring lock group", self.keys)
        for key in self.keys:
            lock = lock_dict.setdefault(key, SharableLock(key))
            if lock.acquire(exclusive=exclusive, block=block, timeout=timeout, check_interval=check_interval, silent=True):
                self.locks[key] = lock
            else:
                self.release()
                ret = False
                break
        return ret

    def release(self):
        # locks release automatically, but we'll force it rather than wait for garbage collection
        print("releasing lock group:", self.keys)
        for lock in self.locks.values():
            lock.release(silent=False)
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
    return lock_dict.setdefault(nkey, SharableLock(nkey))

def info():
    return [(key, lock._is_locked, lock.semaphore.get_value()) for key, lock in lock_dict.items()]

def force_unlock(key):
    try:
        lock_dict.pop(key)
    except KeyError:
        pass


if __name__ == "__main__":

# def server():
    port = 5000
    try:
        port = int(sys.argv[1])
    except IndexError:
        pass
    except:
        raise ValueError("port number not understood", sys.argv[1])
    manager = SyncManager(address=('localhost', port), authkey=b'nbslocking')
    manager.register("get_lock", get_lock)  # , AcquirerProxy
    manager.register("info", info)  # , AcquirerProxy
    manager.register("force_unlock", force_unlock)  # , AcquirerProxy
    manager.register("MultiLock", MultipleLocks)  # , AcquirerProxy
    try:
        server = manager.get_server()
    except OSError:
        print("Another server appears to be running, exiting now")
    else:
        print(manager)
        print('server running', manager.address)
        server.serve_forever()

# server()