import logging
import pathlib
import datetime
import pprint

LOGGER_NAME = 'call_history'


def get_call_logger():
    """ Get the existing call logger object. """
    debug_log = logging.getLogger('call_history')
    return debug_log


def setup_call_logger(root, log_level=logging.DEBUG):
    """ This function sets up the logger for the call history.
    If root is provided and a file has not already been created, it will create a log file in the root directory.
    The filename will be the current date and time.
    """
    debug_log = get_call_logger()
    root_pth = pathlib.Path(root).joinpath("call_history")
    root_pth.mkdir(parents=True, exist_ok=True)
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hdlr_2 = logging.FileHandler(root_pth.joinpath(f"{t}.log"))
    debug_log.setLevel(log_level)
    debug_log.addHandler(hdlr_2)


def log_calls(func):
    """ This is a decorator that logs the function calls and the arguments passed to the function. """
    def wrapper(*args, **kwargs):
        debug_log = get_call_logger()
        debug_log.debug(f"Start Function Call {func.__module__} {func.__name__}")
        debug_log.debug("Args:")
        debug_log.debug(pprint.pformat(args))
        debug_log.debug("KWArgs:")
        debug_log.debug(pprint.pformat(kwargs))
        # for key, val in kwargs.items():
        #     debug_log.debug(f"{key} : {val}")
        ret = func(*args, **kwargs)
        debug_log.debug(f"Exit Function Call {func.__module__} {func.__name__}")
        return ret
    return wrapper


def get_log_path():
    """ Gets the first file handler in the logger and returns the path of the log file. """
    debug_log = get_call_logger()
    for h in debug_log.handlers:
        if isinstance(h, logging.FileHandler):
            return h.baseFilename

