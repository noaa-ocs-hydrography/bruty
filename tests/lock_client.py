from multiprocessing.managers import SyncManager, BaseProxy
from concurrent.futures import ThreadPoolExecutor
import time
from nbs.bruty.nbs_locks import AreaLock, EXCLUSIVE, NON_BLOCKING, LockNotAcquired

def client():
    SyncManager.register('get_lock')
    manager = SyncManager(address=('localhost', 5000), authkey=b'abracadabra')
    manager.connect()
    # distance_time = manager.DistanceTime()
    # distance_time.get_distance_time()
    lck = manager.get_lock((1, 1))
    print("retrieved lock", lck)
    b = lck.acquire()  # block=True, timeout=1)
    print("acquire returned",b)
    if b:
        time.sleep(5)
        lck.release()
        print("lock released")
    else:
        print('try faulty release')
        lck.release()


    #
    # executor = ThreadPoolExecutor(max_workers=3)
    # a = executor.submit(distance_time.get_distance_time)
    # print(a)
    # b = executor.submit(distance_time.get_distance_time)
    # print(b)
    # c = executor.submit(distance_time.get_distance_time)
    # print(c)
    # print(a, b, c)
    # print('done')
    # time.sleep(5)

# client()
def fn(tx, ty):
    return str((tx, ty))
tile_list = [[n,n] for n in range(1000)]

for r in range(1000):
    while 1:
        try:
            print('iteration', r)
            with AreaLock(tile_list, EXCLUSIVE|NON_BLOCKING, fn) as blah:
                print('iteration finished', r)
                break
        except LockNotAcquired:
            print("locked")
            pass
