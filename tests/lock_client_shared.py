from multiprocessing.managers import SyncManager, BaseProxy
from concurrent.futures import ThreadPoolExecutor
import time

def client():
    SyncManager.register('get_lock')
    manager = SyncManager(address=('localhost', 5000), authkey=b'abracadabra')
    manager.connect()
    # distance_time = manager.DistanceTime()
    # distance_time.get_distance_time()
    lck = manager.get_lock((1, 1))
    print("retrieved lock", lck)
    b = lck.acquire(exclusive=False, block=True, timeout=1)
    print("shared acquire returned", b)
    if b:
        time.sleep(5)
        lck.release()
        print("lock released")


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

client()
