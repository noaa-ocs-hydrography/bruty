import configparser
import datetime
import pprint
import io
# import glob
import getpass
import pathlib
import logging
import inspect
import os
# import shutil
import sys
from typing import Union

"""  Sample usage -------
LOGGER = get_logger('xipe.csar.proc')
CONFIG_SECTION = 'combined_raster_processing'

def main():
    do_something()

if __name__ == '__main__':
    run_command_line_configs(main, "Doing Something", CONFIG_SECTION, LOGGER)
    # or just use defaults 
    run_command_line_configs(main)



# or to do the loop manually: 

if __name__ == '__main__':
    if len(sys.argv) > 1:
        use_configs = sys.argv[1:]
    else:
        use_configs = pathlib.Path(__file__).parent.resolve()  # (os.path.dirname(os.path.abspath(__file__))

    warnings = ""
    # LOGGER.info(f'running {len(config_filenames)} configuration(s)')
    for config_filename, config_file in iter_configs(use_configs):
        stringio_warnings = set_stream_logging("xipe", file_level=logging.WARNING, remove_other_file_loggers=False)
        LOGGER.info(f'***************************** Start Run  *****************************')
        LOGGER.info(f'reading "{config_filename}"')
        log_config(config_file, LOGGER)

        config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']
"""


def parse_multiple_values(val: str):
    """ Split a multiline string based on newlines and commas and strip leading/trailing spaces.
    Useful for multiple paths, names and such.

    Parameters
    ----------
    val
        Multiline string that has individual values separated by newlines and/or commas

    Returns
    -------
    list
        Return list of strings parsed from the input value
    """
    lines = val.split("\n")
    full_list = []
    for line in lines:
        names = [name.strip() for name in line.split(",") if name.strip()]
        full_list.extend(names)
    return full_list


def parse_ints_with_ranges(val: str):
    """ Split a multiline string based on newlines and commas and strip leading/trailing spaces.
    Works like declaring page numbers in a print dialog - integers with a dash in between mean a range of integers.

    Parameters
    ----------
    val
        Multiline string that has individual values separated by newlines and/or commas

    Returns
    -------
    list
        Return list of strings parsed from the input value
    """
    lines = val.split("\n")
    full_list = []
    for line in lines:
        for t in line.split(","):
            if t.strip():  # avoid blank lines, two commas next to each other etc.
                try:
                    full_list.append(int(t))
                except ValueError:
                    low, high = t.split("-")
                    full_list.extend(range(int(low), int(high) + 1))  # add one to make the range inclusive, e.g. 3-5 => [3,4,5]
    return full_list


def get_additional_configs(config_filename, base_config_path, parent_paths=tuple()):
    configs = []
    raw_config_file = configparser.ConfigParser()
    found = raw_config_file.read(config_filename)  # read the initial file to get and extra configs and additional defaults
    if len(found) == 0:
        raise FileNotFoundError("File not found " + str(config_filename))

    try:
        extra_confs = [fname.strip() for fname in parse_multiple_values(raw_config_file['DEFAULT']['additional_configs'])]
        extra_confs.reverse()  # file should be in most significant to least - read in the opposite order
    except KeyError:
        extra_confs = []
    # use the parameter base_config_path and if it doesn't exist then see if config_path is set, else use the directory local to the config.
    if base_config_path is None:
        try:
            config_basepath = raw_config_file['DEFAULT']['config_path']
            pth = pathlib.Path(config_basepath)
            if not pth.is_absolute():
                pth = config_filename.parent.joinpath(pth)  # this is an absolute path determined above
            base_config_path = pth
        except KeyError:
            base_config_path = config_filename.parent
    base_config_path = pathlib.Path(base_config_path)  # make sure it's a pathlib Path
    # load each file in the order of priority
    for fname in extra_confs:
        fname = pathlib.Path(fname)
        # if the additional config is an absolute path then use it otherwise use the basepath determined above as the starting point
        if not fname.is_absolute():
            fname = base_config_path.joinpath(fname)
        fpath = pathlib.Path(fname).absolute().resolve()
        if fpath in parent_paths:
            raise FileExistsError("Circular load of configs in " + str(config_filename) + " and " + str(fpath))
        configs.append(fpath)
        configs.extend(get_additional_configs(fpath, base_config_path, parent_paths+(config_filename, )))
    return configs


def load_config(config_filename: Union[str, os.PathLike], base_config_path: Union[str, os.PathLike] = None,
                initial_config: Union[str, os.PathLike, configparser.ConfigParser] = None,
                interp: bool = True):
    """
    Parameters
    ----------
    config_filename
        path to the config INI to load, local to current working directory if no full path is supplied
    base_config_path
        path to additional_configs listed in the config file, if not supplied then local to the config_filename is checked
    initial_config
        values that are to be passed to all the additional
    interp
        specifies if configparser.ExtendedInterpolation will be used.
        If True then special syntax will be evaluated like or the raw string would be returned.

        E.g. if "tablename=pbc_${utm}N" was in the config and in another config utm=19 was specfied then with interp=True
        config['default']['tablename'] would be "pbc_19N" but would return "pbc_${utm}N" if interp was False
    immediate_interp
        specifies if the configs should be immediately evaluated for substitution at every config that is loaded or if evaluation is delayed til all
        configs have been loaded.

        Ex: If config "A" loads sub config "B" which loads sub config "C" which loads sub config "D".
        "D" says utm=18, "C" then says name=${utm}N, "B" then overrides with utm=19 and "A" says last_name=${utm}N.
        If immediate_interp is False then name and last_name will both 19N.
        If immediate_interp is True then last_name will still be 19N but name will be 18N ("C" converts it's ${utm} as soon as it's read and so is 18)
    parent_paths
        This is internal to the recursion to stop circular imports.
        Any addition_configs that are to be loaded but are in the parent_paths will cause a FileExistsError.

    Returns
    -------

    """
    # There are problems with how this was done.
    # If read_dict() is used then the config object is translated and sections have a full copy of data.
    # It an earlier default value will get copied into a section and prevents what should have been a propagated value from a later default.
    # Or a later default can overwrite an earlier section specific value accidentally
    # So all the config filenames with full paths in order should be found and then read() should be called multiple times.
    config_filename = pathlib.Path(config_filename).absolute().resolve()
    extra_configs = get_additional_configs(config_filename, base_config_path)
    use_interp = configparser.ExtendedInterpolation() if interp else None
    config_file = configparser.ConfigParser(interpolation=use_interp)
    if initial_config:
        if isinstance(initial_config, str):
            config_file.read(initial_config)
        else:  # this will accept a dictionary or a configparser instance which acts like a dictionary {section: {key: value}}
            config_file.read_dict(initial_config)
    config_file.read(extra_configs+[config_filename])
    return config_file


def iter_configs(config_filenames: Union[list, str, os.PathLike], log_files: bool = True, default_config_name: Union[str, os.PathLike] = "",
                 base_config_path: Union[str, os.PathLike] = None):
    """ Read all the configs using configparser and optionally modified by a default config file and base_configs.
    A ConfigParser object is created.  Then the default_config_name is loaded, if applicable.
    Then loads all configs from the base_configs directory (local to the script) listed in the [DEFAULT] section 'additional_configs' entry.
    Then looks for a subdirectory of the logged in user (os.getlogin()) and uses that if it exists, otherwise uses the current dir.
    Finally iterates each config in the user subdirectory or current directory so they have the highest priority.

    Parameters
    ----------
    config_filenames
        Either an iterable list of config paths or a str/PathLike that points to a directory to scan.
        If only one config is desired then send it as a list ["c:\\test\\example.config"]
    log_files
        If True then a pair of log files will be created in a log directory in the same path as the config.
        One log gets all message levels the other is Warnings and Errors.
    default_config_name
        The default name to load with each config.  Any info in the current config will take priority over the default values.

    Returns
    -------
    filename str, ConfigParser instance

    """
    user_directory_exists = False
    if isinstance(config_filenames, (str, os.PathLike)):
        use_configs = pathlib.Path(config_filenames)  # (os.path.dirname(os.path.abspath(__file__))
        # Previously, the logged in user was found with os.getlogin(), but this was incompatible with linux systemd services
        # user_dir = use_configs.joinpath(os.getlogin())
        user_dir = use_configs.joinpath(getpass.getuser())
        if user_dir.exists():
            user_directory_exists = True
            use_configs = user_dir
        config_filenames = use_configs.glob('*.config')
    # remove the default names from the list of files to process
    config_filenames = [pathlib.Path(p) for p in config_filenames]
    for config_filename in filter(lambda fname: fname.name != default_config_name, config_filenames):
        config_path, just_cfg_filename = config_filename.parent, config_filename.name
        if base_config_path is None:
            if user_directory_exists:
                base_config_path = config_path.parent.joinpath("base_configs")  # parallel to the user directory, so up one directory from config
            else:
                base_config_path = config_path.joinpath("base_configs")  # local to the config file
            if not base_config_path.exists():
                base_config_path = pathlib.Path("base_configs")  # local to the current directory
                if not base_config_path.exists():
                    base_config_path = None  # use the local directory default

        os.makedirs(os.path.join(config_path, "logs"), exist_ok=True)
        if default_config_name:
            initial_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            initial_config.read(os.path.join(config_path, default_config_name))
        else:
            initial_config = {}
        config_file = load_config(config_filename, base_config_path, initial_config)
        if log_files:
            # sets the parent logger to output to files
            log_level = get_log_level(config_file)
            make_family_of_logs("nbs", os.path.join(config_path, 'logs', just_cfg_filename), log_level=log_level)
        yield config_filename, config_file


def read_config(config_filename: str, **kwargs):
    """ Read a single config file and return the ConfigParser object.
    All other parameters (kwargs) are passed to iter_configs().

    Parameters
    ----------
    config_filename

    Returns
    -------

    """
    for fname, config in iter_configs([config_filename], **kwargs):
        return config


def make_family_of_logs(name, root_filename, log_format=None, remove_other_file_loggers=True, weekly=True, log_level=logging.DEBUG):
    added_handlers = []
    # sets the parent logger to output to files
    if weekly:
        # truncated division by weeks so that we start on the same day of week every time
        # result will be '2022-11-27'
        log_week = datetime.date.fromordinal((datetime.date.today().toordinal() // 7) * 7).isoformat()
        orig_filename = pathlib.Path(root_filename)
        weekly_path = orig_filename.parent.joinpath("logs_" + log_week)
        os.makedirs(weekly_path, exist_ok=True)
        root = weekly_path.joinpath(orig_filename.name)
    if log_level <= logging.DEBUG:
        h1 = set_file_logging(name, str(root) + ".debug.log", log_format=log_format,
                         file_level=logging.DEBUG, remove_other_file_loggers=remove_other_file_loggers)
        added_handlers.extend(h1)
    if log_level <= logging.INFO:
        h2 = set_file_logging(name, str(root) + ".log", log_format=log_format,
                         file_level=logging.INFO, remove_other_file_loggers=False)
        added_handlers.extend(h2)
    if log_level <= logging.WARNING:
        h3 = set_file_logging(name, str(root) + ".warnings.log", log_format=log_format,
                         file_level=logging.WARNING, remove_other_file_loggers=False)
        added_handlers.extend(h3)
    return added_handlers

def close_logs(logger, specific_handlers: list = None, show_open_logs: bool = False):
    """ Close the log files and remove the handlers from the logger.
    This should be called in a pairing with make_family_of_logs() to close the log files and remove the handlers.

    Parameters
    ----------
    logger: str, logging.Logger
        name to pass to logging.getLogger() or a logger object itself to close file and stream handlers from
    specific_handlers
        list of logging.Handler instances to close, if None then all handlers are closed
    show_open_logs
        prints open handlers to the console for all loggers if True

    Returns
    -------

    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    if show_open_logs:  # show all loggers
        for k, v in logging.Logger.manager.loggerDict.items():
            if "nbs" in k:
                print('+ [%s] {%s} ' % (str.ljust(k, 20), str(v.__class__)[8:-2]))
            if not isinstance(v, logging.PlaceHolder):
                for h in v.handlers:
                    print('     +++', str(h.__class__)[8:-2])
    if not specific_handlers:
        specific_handlers = list(logger.handlers)
    for h in specific_handlers:  # cache a list since we will be editing the handlers in the loop
        try:
            if isinstance(h, (logging.FileHandler, logging.StreamHandler)):
                # if h.stream not in (sys.stderr, sys.stdout):
                logger.removeHandler(h)
                h.close()
        except:
            pass  # don't fail because logger already closed or something


def log_config(config_file, logger, absolute=True):
    """ Writes the data from a config file into a log - so the parameters used are stored in the processing log.

    Parameters
    ----------
    config_file
        configparser.ConfigParser instance
    logger
        logging.Logger instance
    absolute
        flag for if variables are evaluated and written with values or as variables like ${var_name}

    Returns
    -------
    None

    """
    if absolute:
        ss = pprint.pformat({section: dict(config_file[section]) for section in config_file.keys()}, width=500)
        logger.info(ss)
    else:  # show with variables in the data e.g.  ${section:variable}
        # archive the config used
        ss = io.StringIO()
        config_file.write(ss)
        ss.seek(0)
        logger.info(ss.read())
        del ss


def get_logger(name: str, log_filename: str = None, file_level: int = None, console_level: int = None, log_format: str = None) -> logging.Logger:
    """ This function creates a top level parent logger for whatever namespace is passed in.
    WARNING, if a parent logger is already setup then this function will exit without modifying it.
    Basically this function works if it's the first thing used in the logging ancestry but not useful if, say

    Up to three loggers are made based on the parameters passed in.
    First a sys.stderr logger is set up (unless the console_level is "logging.NOTSET") for WARNING, ERROR and CRITICAL levels.
    If the console_level is logging.DEBUG or INFO then a sys.stdout is also created.
    When a log_filename is specified then a file handler is also made at the given path.
    
    Formatters are set for all loggers with the definition of '[%(asctime)s] %(name)-15s %(levelname)-8s: %(message)s'
    
    Parameters
    ----------
    name
    log_filename
    file_level
    console_level
    log_format

    Returns
    -------

    """
    if console_level is None:
        console_level = logging.INFO
    logger = logging.getLogger(name)

    # check if logger is already configured
    if logger.level == logging.NOTSET and len(logger.handlers) == 0:
        # check if logger has a parent
        if '.' in name:
            logger.parent = get_logger(name.rsplit('.', 1)[0])
        else:
            # otherwise create a new split-console logger
            logger.setLevel(logging.DEBUG)
            if console_level != logging.NOTSET:
                # this creates a logger which writes DEBUG and INFO to stdout
                # and another that writes WARNING, ERROR, CRITICAL to stderr
                # stderr gets colored differently in pycharm and could be read from a pipe if needed
                # otherwise it doesn't have much effect
                if console_level <= logging.INFO:
                    class LoggingOutputFilter(logging.Filter):
                        def filter(self, rec):
                            return rec.levelno in (logging.DEBUG, logging.INFO)

                    console_output = logging.StreamHandler(sys.stdout)
                    console_output.setLevel(console_level)
                    console_output.addFilter(LoggingOutputFilter())
                    logger.addHandler(console_output)

                console_errors = logging.StreamHandler(sys.stderr)
                console_errors.setLevel(max((console_level, logging.WARNING)))
                logger.addHandler(console_errors)
            # Will create a log file on disk, if a filename is specified
            set_file_logging(name, log_filename, file_level)

    if log_format is None:
        log_format = '[%(asctime)s] %(name)-15s %(levelname)-8s: %(message)s'
    log_formatter = logging.Formatter(log_format)
    for handler in logger.handlers:
        handler.setFormatter(log_formatter)

    return logger


def set_stream_logging(logger_name: str, file_level: int = None, log_format: str = None,
                       remove_other_file_loggers: bool = True):
    io_string = io.StringIO()
    log_stream = logging.StreamHandler(io_string)
    set_file_logging(logger_name, log_stream, file_level, log_format, remove_other_file_loggers)
    return io_string


def set_file_logging(logger_name: str, log_file: Union[str, pathlib.Path, logging.StreamHandler] = None, file_level: int = None,
                     log_format: str = None, remove_other_file_loggers: bool = True):
    added_handlers = []
    logger = logging.getLogger(logger_name)
    logger.debug(f"{logger.name} saving to {log_file}")
    if remove_other_file_loggers:  # remove string and file loggers except for the default stderr, stdout
        for existing_file_handler in [handler for handler in logger.handlers if isinstance(handler, (logging.FileHandler, logging.StreamHandler))]:
            if existing_file_handler.stream not in (sys.stderr, sys.stdout):
                logger.removeHandler(existing_file_handler)
    if log_file is not None:
        if isinstance(log_file, (str, pathlib.Path)):
            file_handler = logging.FileHandler(log_file)
        else:
            file_handler = log_file
        if file_level is None:
            file_level = logging.DEBUG
        file_handler.setLevel(file_level)
        if log_format is None:
            log_format = '[%(asctime)s] %(name)-15s %(levelname)-8s: %(message)s'
        log_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        added_handlers.append(file_handler)
    return added_handlers

default_logger = get_logger("nbs.bruty.scripts")


def convert_to_logging_level(config_level):
    try:
        log_level = int(config_level)
    except ValueError:
        config_level = config_level.upper()
        if config_level == "WARNING":
            log_level = logging.WARNING
        elif config_level == "INFO":
            log_level = logging.INFO
        elif config_level == "DEBUG":
            log_level = logging.DEBUG
        else:
            log_level = logging.WARNING
    return log_level


def get_log_level(config, def_level="WARNING", section="DEFAULT"):
    """ Get a logging level from the config file under the name 'LOG_LEVEL'.
    If the value is an integer then it is used directly.
    If the value is a string then it is converted to an integer based on the logging module's levels.

    Acceptable strings are "WARNING", "INFO", "DEBUG".

    Parameters
    ----------
    config_file
        configparser.ConfigParser or section instance
    def_level
        default logging level if 'LOG_LEVEL' is not in the section
    section
        section of the config file to look in

    Returns
    -------
    int
        logging level (based on logging module's levels like logging.INFO, logging.DEBUG, etc.)

    """
    if isinstance(config, configparser.ConfigParser):
        config_sect = config[section if section in config else 'DEFAULT']
    else:
        config_sect = config
    config_level = config_sect.get('LOG_LEVEL', def_level)
    return convert_to_logging_level(config_level)


def run_configs(func, title, use_configs, section="DEFAULT", logger=default_logger, log_suffix=""):
    """ use_configs as a string denotes a directory to search for all configs.
    use_configs as a list specifies config files to use.
    """
    all_warnings = []

    for config_filename, config_file in iter_configs(use_configs):
        log_level = get_log_level(config_file, section=section)
        make_family_of_logs("nbs", config_filename.parent.joinpath("logs", config_filename.name + "_" + str(os.getpid()) + log_suffix),
                            remove_other_file_loggers=False, log_level=log_level)
        # @TODO expose the stringio_warnings to the caller
        stringio_warnings = set_stream_logging("bruty", file_level=logging.WARNING, remove_other_file_loggers=False)
        logger.info(f'***************************** Start {title}  *****************************')
        logger.info(f'reading "{config_filename}"')
        log_config(config_file, logger)
        config = config_file[section if section in config_file else 'DEFAULT']
        config.logger_warnings = stringio_warnings
        all_warnings.append(stringio_warnings)
        config._source_filename = config_filename
        func(config)
    return all_warnings


def run_command_line_configs(func, title="", section="DEFAULT", logger=default_logger, log_suffix=""):
    """ run configs specified on the command line (sys.argv) or search the directory where func is located (uses inspect module)
    """
    if len(sys.argv) > 1:
        use_configs = sys.argv[1:]
    else:
        use_configs = pathlib.Path(inspect.getfile(func)).parent.resolve()  # (os.path.dirname(os.path.abspath(__file__))
    return run_configs(func, title, use_configs, section, logger, log_suffix=log_suffix)


def show_logger_handlers(logger):
    print("logger handlers--------------------------------------")
    c = logger
    while c:
        for hdlr in c.handlers:
            print(hdlr)
        c = c.parent
    print("end logger handlers--------------------------------------")
