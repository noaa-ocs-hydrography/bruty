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
        for k, v in kwargs.items():
            if callable(v):
                debug_log.debug(str(k) + ": not printable - callable object")
            else:
                debug_log.debug(str(k) + ": " + pprint.pformat(v))
        # for key, val in kwargs.items():
        #     debug_log.debug(f"{key} : {val}")
        ret = func(*args, **kwargs)
        debug_log.debug(f"Exit Function Call {func.__module__} {func.__name__}")
        return ret
    return wrapper


def get_dbg_log_path():
    """ Gets the first file handler in the logger and returns the path of the log file. """
    debug_log = get_call_logger()
    for h in debug_log.handlers:
        if isinstance(h, logging.FileHandler):
            return h.baseFilename


if __name__ == '__main__':
    import pyarrow.parquet as pq
    import geopandas as gpd
    import time, json

    t = time.time()
    path_to_survey_data = r"V:\NBS_Data_Qualified\PBD_California_UTM11N_MLLW\NOAA_Direct_OCS\ESD\Original\E01008\E01008_MB_MLLW_1of2.parquet"
    schema_dict = {field.name: str(field.type) for field in pq.read_schema(path_to_survey_data)}
    UNCERTAINTY = 'Uncertainty'
    GEOMETRY = 'geometry'
    CLASSIFICATION = 'Classification'
    expected_fields = [UNCERTAINTY, CLASSIFICATION, GEOMETRY]
    missing_fields = [field for field in expected_fields if field not in schema_dict]

    # stage the file for reading
    parquet_file = pq.ParquetFile(path_to_survey_data)

    # use crs from file if found in the "primary column" metadata
    metadata = parquet_file.metadata
    geo_metadata = json.loads(metadata.metadata[b'geo'])
    primary_column = geo_metadata.get('primary_column', None)
    srs = geo_metadata.get('columns', {}).get(primary_column, {}).get('crs', {})
    wkt = gpd.GeoDataFrame(columns=[GEOMETRY]).set_crs(srs).crs.to_wkt()

    # read the file into dataframe batches yielding numpy arrays
    for batch in parquet_file.iter_batches(batch_size=10, columns=expected_fields):
        df = batch.to_pandas()
        pts = gpd.GeoSeries.from_wkb(df[GEOMETRY])
        x = pts.x.values
        y = pts.y.values
        depth = pts.z.values
        uncertainty = df[UNCERTAINTY].values
        print(wkt, x, y, depth, uncertainty)
        break
    print(time.time() - t)
