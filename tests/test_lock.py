from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock

port = 5000
use_locks(port)

def test_too_fast():
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

if __name__ == "__main__":
    test_too_fast()
