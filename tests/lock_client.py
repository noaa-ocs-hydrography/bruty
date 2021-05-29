from multiprocessing.managers import SyncManager, BaseProxy
from concurrent.futures import ThreadPoolExecutor
import time

from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, start_server, \
    current_address

# As of Windows 10
# First start a lock_server (nbs.bruty.lockserver.py) on port 5000
# then run this file four separate times (4 processes) and eventually it will overrun the sockets and it'll start the wait timeout.

start_server(5000)


def fn(tx, ty):
    return str((tx, ty))
tile_list = [[n,n] for n in range(2)]

for r in range(10000):
    while 1:
        try:
            print('iteration', r)
            with AreaLock(tile_list, EXCLUSIVE|NON_BLOCKING, fn) as blah:
                print('iteration finished', r)
                break
        except LockNotAcquired:
            print("locked")
            pass
