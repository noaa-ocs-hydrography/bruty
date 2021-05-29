""" Using portalocker as a cheap test at first, it is a cross platform file locking library that relies on the OS.
Later we can move to a postgres or redis more robust locking system.  portalocker does have some sort of redis support.
"""
import os
import pathlib
import random
import functools
import sys
import time
import win32con
import win32file
import pywintypes
import winerror
import msvcrt
import enum
from multiprocessing.managers import SyncManager


class LockNotAcquired(Exception):
    pass

class SqlLock:
    """
    """
    def __init__(self, conn):
        """
        Parameters
        ----------
        conn
        """
        self.conn = conn

    def __enter__(self):
        self.conn.isolation_level = 'EXCLUSIVE'
        self.conn.execute('BEGIN EXCLUSIVE')
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.isolation_level = 'DEFERRED'


class UninitializedManager:
    def __init__(self):
        pass

    def __getattr__(self, item):
        raise RuntimeError("SyncManager not started, you must call start_server(port) before trying to use locks")


manager = UninitializedManager()

EXCLUSIVE = 1
SHARED = 2
NON_BLOCKING = 4


def start_server(port=5000):
    """
    Parameters
    ----------
    port

    Returns
    -------

    """
    global manager

    try:
        manager = SyncManager(address=('localhost', port), authkey=b'nbslocking')
        while 1:
            try:
                manager.connect()
                break
            except OSError:  # the connection may not be finished closing?
                print("oserr - retrying")
                time.sleep(.5)
    except ConnectionRefusedError as e:
        print("\nCould not connect to the locking server, is it running and is the connection url+port correct?\n\n")
        raise e
    manager.register('get_lock')
    manager.register("MultiLock")
    manager.register('info')
    manager.register("force_unlock")


def patient_lock(func):
    """ There is a windows limit for the number of connections and how fast they open/close.
    If this limit is exceeded then an OSError is raised.
    This decorator will wait up to ~4 minutes to connect and otherwise raise an OSError
    """
    @functools.wraps(func)
    def wrap(*args, **kywrds):
        for delay in range(-3, 9):
            try:
                if delay > -3:
                    time.sleep(2**delay)
                lock = func(*args, **kywrds)
                break
            except TypeError as e:
                print(f"TypeError in communication - normally this is a closed handle, retying in {2 ** (delay + 1)} seconds")
            except OSError as e:  # the connection may not be finished closing?
                print(f"OSError - delaying to let socket server catch up then retrying in {2 ** (delay + 1)} seconds")
        try:
            return lock
        except NameError:
            raise OSError("couldn't connect to lock server")
    return wrap

if True:
    @patient_lock
    def get_lock(*args, **kywrds):
        lock = manager.get_lock(*args, **kywrds)
        lock.acquire = patient_lock(lock.acquire)
        lock.release = patient_lock(lock.release)
        return lock


    @patient_lock
    def get_multilock(*args, **kywrds):
        lock = manager.MultiLock(*args, **kywrds)
        lock.acquire = patient_lock(lock.acquire)
        lock.release = patient_lock(lock.release)
        return lock
else:
    def get_lock(*args, **kywrds):
        lock = manager.get_lock(*args, **kywrds)
        return lock


    def get_multilock(*args, **kywrds):
        lock = manager.MultiLock(*args, **kywrds)
        return lock


def current_address():
    return manager.address

def get_info():
    return manager.info()

def force_unlock(key):
    return manager.force_unlock(key)

class BaseLock:  # this is an abstract class -- need to derive and get your own lock
    def __init__(self, flags=EXCLUSIVE|NON_BLOCKING, timeout=-1):
        """
        Parameters
        ----------
        flags
        timeout
        """
        self.exclusive = not (flags & SHARED)
        self.block = not (flags & NON_BLOCKING)
        self.timeout = timeout
        self.acquired = False

    def acquire(self):
        """
        Returns
        -------

        """
        # randomize the check interval to keep two processes from checking at the same time.
        if not self.acquired:
            ret = self.lock.acquire(self.exclusive, self.block, self.timeout)
            self.acquired = ret
        if not self.acquired:
            raise LockNotAcquired(f"Failed to acuire")
        return self.acquired

    def release(self):
        if self.acquired:
            self.lock.release()
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


class FileLock(BaseLock):
    def __init__(self, fname, mode='r+', flags=EXCLUSIVE|NON_BLOCKING, timeout=-1):
        """
        Parameters
        ----------
        fname
        mode
        flags
        timeout
        """
        super().__init__(flags=flags, timeout=timeout)
        self.mode = mode
        self.fname = fname
        self.lock = get_lock(self.fname)
        self._fh = None

    def openfile(self):
        return open(self.fname, self.mode)

    @property
    def fh(self):
        # make this to mimic the portalocker implementation
        if self._fh is None:
            self._fh = self.openfile()
        return self._fh

    def __enter__(self):
        if self.acquire():
            try:
                handle = self.openfile()
            except (FileNotFoundError, PermissionError) as e:
                self.release()
                raise e
        else:
            raise LockNotAcquired(f"Failed to acquire lock on {self.fname}")
        return handle



class TileLock(FileLock):
    def __init__(self, tx, ty, idn, flags, conv_txy_to_path):
        """
        Parameters
        ----------
        tx
        ty
        idn
        flags
        conv_txy_to_path
        """
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


# try this multilock idea to reduce the number of socket connections that result from calling many single lock instances
class AreaLock(BaseLock):
    def __init__(self, tile_list, flags, conv_txy_to_path, sid=None, timeout=-1):
        """
        Parameters
        ----------
        tile_list
        flags
        conv_txy_to_path
        sid
        timeout
        """
        super().__init__(flags=flags, timeout=timeout)
        self.tile_list = tile_list
        keys = [conv_txy_to_path(tx, ty) for tx, ty in self.tile_list]
        self.lock = get_multilock(keys)
        # print(self.lock.get_keys())
        self.sid = sid


# Locks based on a string, does not need to be an actual file path
class NameLock(BaseLock):
    def __init__(self, name, mode='r+', flags=EXCLUSIVE|NON_BLOCKING, timeout=-1):
        """
        Parameters
        ----------
        name
        mode
        flags
        timeout
        """
        super().__init__(flags=flags, timeout=timeout)
        self.mode = mode
        self.name = name
        self.lock = get_lock(self.name)


# The code below is the equivalent of portalocker but for windows only and uses msvcrt (which might not exist in Python 3.6-3.8?)
# basically fills the need for a os level file lock, not like the lock server being used above.

LOCK_EX = 0x1  #: exclusive lock
LOCK_SH = 0x2  #: shared lock
LOCK_NB = 0x4  #: non-blocking
LOCK_UN = msvcrt.LK_UNLCK  #: unlock


class LockFlags(enum.IntFlag):
    EXCLUSIVE = LOCK_EX  #: exclusive lock
    SHARED = LOCK_SH  #: shared lock
    NON_BLOCKING = LOCK_NB  #: non-blocking
    UNBLOCK = LOCK_UN  #: unlock


__overlapped = pywintypes.OVERLAPPED()
lock_length = int(2**31 - 1)
current_time = getattr(time, "monotonic", time.time)
DEFAULT_TIMEOUT = 5
DEFAULT_CHECK_INTERVAL = 0.25
LOCK_METHOD = LockFlags.EXCLUSIVE | LockFlags.NON_BLOCKING


class BaseLockException(Exception):
    # Error codes:
    LOCK_FAILED = 1

    def __init__(self, *args, fh=None, **kwargs):
        self.fh = fh
        Exception.__init__(self, *args, **kwargs)


class LockException(BaseLockException):
    pass


class AlreadyLocked(BaseLockException):
    pass


class FileToLarge(BaseLockException):
    pass


def file_lock(file_, flags: LockFlags):
    if flags & LockFlags.SHARED:
        if sys.version_info.major == 2:
            if flags & LockFlags.NON_BLOCKING:
                mode = win32con.LOCKFILE_FAIL_IMMEDIATELY
            else:
                mode = 0

        else:
            if flags & LockFlags.NON_BLOCKING:
                mode = msvcrt.LK_NBRLCK
            else:
                mode = msvcrt.LK_RLCK

        # is there any reason not to reuse the following structure?
        hfile = win32file._get_osfhandle(file_.fileno())
        try:
            win32file.LockFileEx(hfile, mode, 0, -0x10000, __overlapped)
        except pywintypes.error as exc_value:
            # error: (33, 'LockFileEx', 'The process cannot access the file
            # because another process has locked a portion of the file.')
            if exc_value.winerror == winerror.ERROR_LOCK_VIOLATION:
                raise LockException(
                    LockException.LOCK_FAILED,
                    exc_value.strerror,
                    fh=file_)
            else:
                # Q:  Are there exceptions/codes we should be dealing with
                # here?
                raise
    else:
        if flags & LockFlags.NON_BLOCKING:
            mode = msvcrt.LK_NBLCK
        else:
            mode = msvcrt.LK_LOCK

        # windows locks byte ranges, so make sure to lock from file start
        try:
            savepos = file_.tell()
            if savepos:
                # [ ] test exclusive lock fails on seek here
                # [ ] test if shared lock passes this point
                file_.seek(0)
                # [x] check if 0 param locks entire file (not documented in
                #     Python)
                # [x] fails with "IOError: [Errno 13] Permission denied",
                #     but -1 seems to do the trick

            try:
                msvcrt.locking(file_.fileno(), mode, lock_length)
            except IOError as exc_value:
                # [ ] be more specific here
                raise LockException(
                    LockException.LOCK_FAILED,
                    exc_value.strerror,
                    fh=file_)
            finally:
                if savepos:
                    file_.seek(savepos)
        except IOError as exc_value:
            raise LockException(
                LockException.LOCK_FAILED, exc_value.strerror,
                fh=file_)

def file_unlock(file_):
    try:
        savepos = file_.tell()
        if savepos:
            file_.seek(0)

        try:
            msvcrt.locking(file_.fileno(), LockFlags.UNBLOCK,
                           lock_length)
        except IOError as exc:
            exception = exc
            if exc.strerror == 'Permission denied':
                hfile = win32file._get_osfhandle(file_.fileno())
                try:
                    win32file.UnlockFileEx(
                        hfile, 0, -0x10000, __overlapped)
                except pywintypes.error as exc:
                    exception = exc
                    if exc.winerror == winerror.ERROR_NOT_LOCKED:
                        # error: (158, 'UnlockFileEx',
                        #         'The segment is already unlocked.')
                        # To match the 'posix' implementation, silently
                        # ignore this error
                        pass
                    else:
                        # Q:  Are there exceptions/codes we should be
                        # dealing with here?
                        raise
            else:
                raise LockException(
                    LockException.LOCK_FAILED,
                    exception.strerror,
                    fh=file_)
        finally:
            if savepos:
                file_.seek(savepos)
    except IOError as exc:
        raise LockException(
            LockException.LOCK_FAILED, exc.strerror,
            fh=file_)



class Lock:

    def __init__(
            self,
            filename,
            mode: str = 'a',
            timeout: float = DEFAULT_TIMEOUT,
            check_interval: float = DEFAULT_CHECK_INTERVAL,
            fail_when_locked: bool = False,
            flags: LockFlags = LOCK_METHOD, **file_open_kwargs):
        """Lock manager with build-in timeout
        filename -- filename
        mode -- the open mode
        timeout -- timeout when trying to acquire a lock
        check_interval -- check interval while waiting
        fail_when_locked -- after the initial lock failed, return an error
            or lock the file
        **file_open_kwargs -- The kwargs for the `open(...)` call
        fail_when_locked is useful when multiple threads/processes can race
        when creating a file. If set to true then the system will wait till
        the lock was acquired and then return an AlreadyLocked exception.
        """

        if 'w' in mode:
            truncate = True
            mode = mode.replace('w', 'a')
        else:
            truncate = False

        self.fh = None
        self.filename: str = str(filename)
        self.mode: str = mode
        self.truncate: bool = truncate
        self.timeout: float = timeout
        self.check_interval: float = check_interval
        self.fail_when_locked: bool = fail_when_locked
        self.flags: LockFlags = flags
        self.file_open_kwargs = file_open_kwargs

    def acquire(
            self, timeout: float = None, check_interval: float = None,
            fail_when_locked: bool = None):
        """Acquire the locked filehandle"""
        if timeout is None:
            timeout = self.timeout
        if timeout is None:
            timeout = 0

        if check_interval is None:
            check_interval = self.check_interval

        if fail_when_locked is None:
            fail_when_locked = self.fail_when_locked

        # If we already have a filehandle, return it
        fh = self.fh
        if fh:
            return fh

        # Get a new filehandler
        fh = self._get_fh()

        def try_close():  # pragma: no cover
            # Silently try to close the handle if possible, ignore all issues
            try:
                fh.close()
            except Exception:
                pass

        # Try till the timeout has passed
        timeout_end = current_time() + timeout
        exception = None
        while timeout_end > current_time():
            try:
                # Try to lock
                fh = self._get_lock(fh)
                break
            except LockException as exc:
                # Python will automatically remove the variable from memory
                # unless you save it in a different location
                exception = exc

                # We already tried to the get the lock
                # If fail_when_locked is True, stop trying
                if fail_when_locked:
                    try_close()
                    raise AlreadyLocked(exception)

                # Wait a bit
                time.sleep(check_interval)

        else:
            try_close()
            # We got a timeout... reraising
            raise LockException(exception)

        # Prepare the filehandle (truncate if needed)
        fh = self._prepare_fh(fh)

        self.fh = fh
        return fh

    def release(self):
        """Releases the currently locked file handle"""
        if self.fh:
            file_unlock(self.fh)
            self.fh.close()
            self.fh = None

    def _get_fh(self):
        """Get a new filehandle"""
        return open(self.filename, self.mode, **self.file_open_kwargs)

    def _get_lock(self, fh):
        """
        Try to lock the given filehandle
        returns LockException if it fails"""
        file_lock(fh, self.flags)
        return fh

    def _prepare_fh(self, fh):
        """
        Prepare the filehandle for usage
        If truncate is a number, the file will be truncated to that amount of
        bytes
        """
        if self.truncate:
            fh.seek(0)
            fh.truncate(0)

        return fh

    def __enter__(self):
        return self.acquire()

    def __exit__(self, type_, value, tb):
        self.release()

    def __delete__(self, instance):
        instance.release()

