import pytest
import os

from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, Lock, LockFlags, AlreadyLocked


def test_too_fast():
    # port = 5000
    # use_locks(port)
    path = "c:\\temp\\test.lock"
    lock = FileLock(path)
    if lock.acquire():
        for n in range(100000):
            fail_lock = FileLock(path)
            try:
                assert not fail_lock.acquire()
            except TypeError as e:
                assert False
            except LockNotAcquired:
                print(n)


def test_2_shared_locks():
    fname = "shared.lock"
    with Lock(fname, fail_when_locked=True, flags=LockFlags.SHARED|LockFlags.NON_BLOCKING) as lock1:
        lock2 = Lock(fname, fail_when_locked=True, flags=LockFlags.SHARED|LockFlags.NON_BLOCKING)
        lock2.acquire()
        lock2.release()
    os.remove(fname)


def test_shared_then_exclusive():
    fname = "shared.lock"
    with Lock(fname, fail_when_locked=True, flags=LockFlags.SHARED|LockFlags.NON_BLOCKING) as lock1:
        try:
            lock2 = Lock(fname, fail_when_locked=True, flags=LockFlags.EXCLUSIVE|LockFlags.NON_BLOCKING)
            lock2.acquire()
            lock2.release()
        except AlreadyLocked:
            pass
        else:
            raise Exception("AlreadyLocked should have been raised")
    os.remove(fname)


def test_exclusive_then_shared():
    fname = "shared.lock"
    with Lock(fname, fail_when_locked=True, flags=LockFlags.EXCLUSIVE|LockFlags.NON_BLOCKING) as lock1:
        try:
            lock2 = Lock(fname, fail_when_locked=True, flags=LockFlags.SHARED|LockFlags.NON_BLOCKING)
            lock2.acquire()
            lock2.release()
        except AlreadyLocked:
            pass
        else:
            raise Exception("AlreadyLocked should have been raised")
    os.remove(fname)


def test_exclusive_twice_with():
    fname = "shared.lock"
    with Lock(fname, fail_when_locked=True, flags=LockFlags.EXCLUSIVE|LockFlags.NON_BLOCKING) as lock1:
        try:
            with Lock(fname, fail_when_locked=True, flags=LockFlags.EXCLUSIVE | LockFlags.NON_BLOCKING) as lock2:
                raise Exception("Should not reach here")
        except AlreadyLocked:
            pass
    os.remove(fname)


def test_shared_then_exclusive_blocking():
    fname = "shared.lock"
    with Lock(fname, fail_when_locked=True, flags=LockFlags.SHARED|LockFlags.NON_BLOCKING) as lock1:
        try:
            lock2 = Lock(fname, fail_when_locked=True, flags=LockFlags.EXCLUSIVE)
            lock2.acquire()
            lock2.release()
        except AlreadyLocked:
            pass
        else:
            raise Exception("AlreadyLocked should have been raised")
    os.remove(fname)


def test_file_limit():
    root = "test_max_files"
    os.makedirs(root, exist_ok=True)
    d = {}
    for n in range(10000):  # failed at 8180 on NBS03
        fname = root + f"\\lock.{n}"
        lock = Lock(fname, fail_when_locked=True, flags=LockFlags.EXCLUSIVE)
        try:
            lock.acquire()
        except OSError:
            break
        d[n] = [fname, lock]
    for fname, lock in d.values():
        lock.release()
        os.remove(fname)


if __name__ == "__main__":
    print("shared_then_shared")
    test_2_shared_locks()
    print("shared_then_exclusive")
    test_shared_then_exclusive()
    print("exclusive_then_shared")
    test_exclusive_then_shared()
    print("exclusive_twice_with")
    test_exclusive_twice_with()
    print("shared_then_exclusive_blocking")
    test_shared_then_exclusive_blocking()
