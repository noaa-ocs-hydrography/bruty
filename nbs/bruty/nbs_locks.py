""" Using portalocker as a cheap test at first, it is a cross platform file locking library that relies on the OS.
Later we can move to a postgres or redis more robust locking system.  portalocker does have some sort of redis support.
"""
import os
import pathlib
import random
import time

use_multiprocessing = True
use_portalocker = False


class LockNotAcquired(Exception):
    pass


if use_multiprocessing:
    from multiprocessing.managers import SyncManager

    EXCLUSIVE = 1
    SHARED = 2
    NON_BLOCKING = 4

    try:
        manager = SyncManager(address=('localhost', 5000), authkey=b'nbslocking')
        while 1:
            try:
                manager.connect()
                break
            except OSError as e:  # the connection may not be finished closing?
                print("oserr - retrying")
                time.sleep(.5)
    except ConnectionRefusedError as e:
        print("\nCould not connect to the locking server, is it running and is the connection url+port correct?\n\n")
        raise e
    manager.register('get_lock')
    manager.register("MultiLock")

    class BaseLock:  # this is an abstract class -- need to derive and get your own lock
        def __init__(self, flags=EXCLUSIVE|NON_BLOCKING, timeout=-1):
            self.exclusive = not (flags & SHARED)
            self.block = not (flags & NON_BLOCKING)
            self.timeout = timeout
            self.acquired = False


        def acquire(self):
            # randomize the check interval to keep two processes from checking at the same time.
            if not self.acquired:
                while 1:
                    try:
                        ret = self.lock.acquire(self.exclusive, self.block, self.timeout)
                        break
                    except OSError as e:  # the connection may not be finished closing?
                        print("oserr - retrying")
                        time.sleep(.5)
                self.acquired = ret
            if not self.acquired:
                raise LockNotAcquired(f"Failed to acuire")
            return self.acquired

        def release(self):
            if self.acquired:
                while 1:
                    try:
                        self.lock.release()
                        break
                    except OSError as e:  # the connection may not be finished closing?
                        print("oserr - retrying")
                        time.sleep(.5)
                self.acquired = False

        def __del__(self):
            self.release()

        def notify(self):
            pass

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()

    class Lock(BaseLock):
        def __init__(self, fname, mode='r+', flags=EXCLUSIVE|NON_BLOCKING, timeout=-1):
            super().__init__(flags=flags, timeout=timeout)
            self.mode = mode
            self.fname = fname
            while 1:
                try:
                    self.lock = manager.get_lock(self.fname)
                    break
                except OSError as e:  # the connection may not be finished closing?
                    print("oserr - retrying")
                    time.sleep(.5)

        def openfile(self):
            return open(self.fname, self.mode)

        def __enter__(self):
            if self.acquire():
                handle = self.openfile()
            else:
                raise LockNotAcquired(f"Failed to acquire lock on {self.fname}")
            return handle



    class TileLock(Lock):
        def __init__(self, tx, ty, idn, flags, conv_txy_to_path):
            fname = conv_txy_to_path(tx, ty)
            no_block_flags = flags | NON_BLOCKING
            super().__init__(fname, 'r', no_block_flags)

        def release(self):
            try:
                self.lock  # make sure the lock was made.  An exception in TileLock.__init__ makes .lock not exist.
            except AttributeError:
                pass
            else:
                super().release()


    # class ReadTileLock(Lock):  # actively being read, if fails then add to the waiting to read lock
    #     pass
    # class WriteTileLock(Lock):  # actively being modified, if fails then add to the waiting to write lock
    #     pass
    # class PendingReadLock(WriteLock):  # something wants to read but there are active/pending writes
    #     pass
    # class PendingWriteLock(WriteLock):  # something wants to modify but there are active reads
    #     pass


    class AreaLock1:
        def __init__(self, tile_list, flags, conv_txy_to_path, sid=None):
            self.locks = []
            self.tile_list = tile_list
            self.sid = sid
            self.flags = flags
            self.conv_txy_to_path = conv_txy_to_path

        def acquire(self):
            try:
                print(f"Trying to lock {len(self.tile_list)} tiles")
                for tx, ty in self.tile_list:
                    self.locks.append(TileLock(tx, ty, self.sid, self.flags, self.conv_txy_to_path))
                    self.locks[-1].acquire()
            except (LockNotAcquired,):
                self.release()
                raise LockNotAcquired(f"Failed to acquire lock on {self.conv_txy_to_path}")
            return True

        def release(self):
            # locks release automatically, but we'll force it rather than wait for garbage collection
            for lock in self.locks:
                lock.release()
            self.locks = []

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()

        def __del__(self):
            self.release()

    # try this multilock idea to reduce the number of socket connections that result from calling many single lock instances
    class AreaLock(BaseLock):
        def __init__(self, tile_list, flags, conv_txy_to_path, sid=None, timeout=-1):
            super().__init__(flags=flags, timeout=timeout)
            self.tile_list = tile_list
            keys = [conv_txy_to_path(tx, ty) for tx, ty in self.tile_list]
            self.lock = manager.MultiLock(keys)
            # print(self.lock.get_keys())
            self.sid = sid


elif use_portalocker:
    import portalocker

    EXCLUSIVE = portalocker.constants.LockFlags.EXCLUSIVE
    SHARED = portalocker.constants.LockFlags.SHARED
    NON_BLOCKING = portalocker.constants.LockFlags.NON_BLOCKING


    class Lock:
        def __init__(self, fname, mode, flags, fail_when_locked=False):
            self.lock = portalocker.Lock(fname, mode=mode, timeout=120, fail_when_locked=fail_when_locked, flags=flags)

        def acquire(self):
            # randomize the check interval to keep two processes from checking at the same time.
            return self.lock.acquire(check_interval=1+random.randrange(0,10)/10.0)

        def release(self):
            self.lock.release()

        def is_active(self):
            return self.lock.fh is not None

        def __del__(self):
            self.release()

        @property
        def fh(self):
            return self.lock.fh

        def notify(self):
            pass

        def __enter__(self):
            return self.acquire()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()


    class TileLock(Lock):
        def __init__(self, tx, ty, idn, flags, conv_txy_to_path):
            parent = conv_txy_to_path(tx, ty)
            fname = pathlib.Path(parent).joinpath('lock.in_use')
            if not os.path.exists(fname):
                os.makedirs(parent, exist_ok=True)
                f = open(fname, 'w')
                f.close()
            super().__init__(fname, 'r', flags, fail_when_locked=True)

        def release(self):
            try:
                self.lock.lock  # make sure the lock was made.  An exception in TileLock.__init__ makes .lock not exist.
            except AttributeError:
                pass
            else:
                super().release()


    # class ReadTileLock(Lock):  # actively being read, if fails then add to the waiting to read lock
    #     pass
    # class WriteTileLock(Lock):  # actively being modified, if fails then add to the waiting to write lock
    #     pass
    # class PendingReadLock(WriteLock):  # something wants to read but there are active/pending writes
    #     pass
    # class PendingWriteLock(WriteLock):  # something wants to modify but there are active reads
    #     pass


    class AreaLock:
        def __init__(self, tile_list, flags, conv_txy_to_path, sid=None):
            self.locks = []
            self.tile_list = tile_list
            self.sid = sid
            self.flags = flags
            self.conv_txy_to_path = conv_txy_to_path

        def acquire(self):
            try:
                print(f"Trying to lock {len(self.tile_list)} tiles")
                for tx, ty in self.tile_list:
                    self.locks.append(TileLock(tx, ty, self.sid, self.flags, self.conv_txy_to_path))
                    self.locks[-1].acquire()
            except (portalocker.exceptions.LockException, portalocker.exceptions.AlreadyLocked):
                self.release()
                raise LockNotAcquired(f"Failed to acquire lock on {self.conv_txy_to_path}")
            except OSError:
                self.release()
                raise LockNotAcquired(f"Trying to lock too many files, need to migrate to postgres")
            return True

        def release(self):
            # locks release automatically, but we'll force it rather than wait for garbage collection
            for lock in self.locks:
                lock.release()
            self.locks = []

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()

        def __del__(self):
            self.release()

