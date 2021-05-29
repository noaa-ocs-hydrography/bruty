import os
import io
import sys
import pathlib
import subprocess
import logging

from nbs.configs import get_logger, iter_configs, set_stream_logging, log_config, parse_multiple_values
from nbs.bruty.nbs_postgres import get_nbs_records, connect_params_from_config

# Caris python way of converting from csar
import fuse_dev
from fuse_dev.scripts.proc_io.fix_csar import generate_argument_string, execute_subprocess
from fuse_dev.fuse.proc_io.proc_io import ProcIO

LOGGER = get_logger('nbs.bruty.convert_csar')
CONFIG_SECTION = 'convert_csar'


def convert_csar_python(csar_path, metadata):
    csar_path = pathlib.Path(csar_path)
    path_to_fuse_dev = pathlib.Path(fuse_dev.__path__[0]).parent
    old_script = str(path_to_fuse_dev.joinpath(r'fuse_dev\scripts\csar_convert\csar_to_temp.py'))
    new_script = pathlib.Path(__file__).parent.joinpath("caris_env_csar_convert.py")
    command = generate_argument_string(str(new_script), pythonpath=f'{path_to_fuse_dev}')
    full_command = command + " " + str(csar_path)
    execute_subprocess(full_command, override_logger=LOGGER)

    elev_path = csar_path.with_suffix('.elev.raster.npy')
    uncert_path = csar_path.with_suffix('.uncrt.raster.npy')
    geotransform_path = csar_path.with_suffix('.geotransform.txt')

    if elev_path.exists():  # see if '.raster.npy' was made and convert to tif
        tif_path = csar_path.with_suffix('.csar.tif')
        output_path = None
        if geotransform_path.exists() and elev_path.exists() and uncert_path.exists():
            tif_writer = ProcIO('npy', 'tif')
            temp_files = (str(elev_path), str(uncert_path), str(geotransform_path))
            tif_writer.write(temp_files, str(tif_path), metadata)

            if tif_path.exists():
                # delete elevation band npy, uncertainty band npy, and geotransform file,
                # they are temp files
                os.remove(elev_path)
                os.remove(uncert_path)
                os.remove(geotransform_path)
                output_path = tif_path
            else:
                print('Raster Csar Error (npy to tif): ', csar_path)
        else:
            print('Raster Csar Error (csar to npy): ', csar_path)
            if geotransform_path.exists():
                os.remove(geotransform_path)
            if os.path.exists(elev_path):
                os.remove(elev_path)
            if os.path.exists(uncert_path):
                os.remove(uncert_path)

    elif csar_path.with_suffix('.bruty.npz').exists():
        output_path = csar_path.with_suffix('.bruty.npz')
    else:
        output_path = None
    return str(output_path) if output_path else None


def convert_csar_carisbatch(carisbatch, epsg, conn_info, use_zip=False, dest_path=None,
                             use_never_post_flag=True):
    """Quick script that converts CSAR data using Caris' carisbatch.exe to convert to bag or xyz points"""

    for table_name in conn_info.tablenames:
        fields, records = get_nbs_records(table_name, conn_info)

        for cnt, record in enumerate(records):
            fname = record['manual_to_filename']
            if fname is None or not fname.strip():
                fname = record['script_to_filename']
            if fname is not None and fname.strip().lower().endswith("csar"):
                if dest_path is not None:
                    local_fname = fname.lower().replace('\\\\nos.noaa\\OCS\\HSD\\Projects\\NBS\\NBS_Data'.lower(), dest_path)
                else:
                    local_fname = fname
                # if use_for_navigation_flag and not record['for_navigation']:
                #     continue
                if use_never_post_flag and record['never_post']:
                    continue
                if not os.path.exists(f"{local_fname}.csv.zip") and not os.path.exists(f"{local_fname}.csv") and \
                   not os.path.exists(f"{local_fname}.depth.tif") and not os.path.exists(f"{local_fname}.elev.tif"):
                    print("processing", record['nbs_id'], cnt, table_name, local_fname)
                    print(record['manual_to_filename'], record['script_to_filename'])
                    convert_file(carisbatch, fname, local_fname, epsg, use_zip)


def convert_file(carisbatch, src_fname, dest_fname, epsg, use_zip=False):
                    cmd = f'"{carisbatch}" -r ExportRaster --output-format GEOTIFF --compression LZW --include-band Depth --include-band Uncertainty "{src_fname}" "{dest_fname}.depth.tif"'
                    p = subprocess.Popen(cmd)
                    p.wait()
                    if not os.path.exists(f"{dest_fname}.depth.tif"):
                        cmd = f'"{carisbatch}" -r ExportRaster --output-format GEOTIFF --compression LZW --include-band Elevation --include-band Uncertainty "{src_fname}" "{dest_fname}.elev.tif"'
                        p = subprocess.Popen(cmd)
                        p.wait()
                        if not os.path.exists(f"{dest_fname}.elev.tif"):
                            cmd = f'"{carisbatch}" -r exportcoveragetoascii --include-band Depth 3 --include-band Uncertainty 3 --output-crs EPSG:{epsg} --coordinate-format GROUND --coordinate-precision 2 --coordinate-unit m "{src_fname}" "{dest_fname}.csv"'
                            p = subprocess.Popen(cmd)
                            p.wait()
                            if os.path.exists(f"{dest_fname}.csv"):
                                if use_zip:
                                    p = subprocess.Popen(f'python -m zipfile -c "{dest_fname}.csv.zip" "{dest_fname}.csv"')
                                    p.wait()
                                    os.remove(f'{dest_fname}.csv')
                                print("was points")
                            else:
                                print("failed as points and raster????????????????????")
def run_configs():
    if len(sys.argv) > 1:
        use_configs = sys.argv[1:]
    else:
        use_configs = pathlib.Path(__file__).parent.resolve()  # (os.path.dirname(os.path.abspath(__file__))

    warnings = ""
    for config_filename, config_file in iter_configs(use_configs):
        stringio_warnings = set_stream_logging("bruty", file_level=logging.WARNING, remove_other_file_loggers=False)
        LOGGER.info(f'***************************** Start Run  *****************************')
        LOGGER.info(f'reading "{config_filename}"')
        log_config(config_file, LOGGER)

        config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']
        epsg = pathlib.Path(config['epsg'])

        tablenames, database, hostname, port, username, password = connect_params_from_config(config)
        caris_batch_path = config['carisbatch']
        convert_csar(caris_batch_path, epsg, tablenames, database, username, password, hostname, port)

def main():
    convert_csar_python(sys.argv[1])
    # r'\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBD_LALB_UTM11_MLLW\NOAA_NCEI_OCS\BAGs\Manual\W00430_MB_64m_MLLW_4of5_upsample.csar')

if __name__ == '__main__':

    # default_config_name = "default.config"

    # turn prints into logger messages
    orig_print = print
    def print(*args, **kywds):
        f = io.StringIO()
        ky = kywds.copy()
        ky['file'] = f
        orig_print(*args, **ky)  # build the string
        LOGGER.info(f.getvalue()[:-1])  # strip the newline at the end
    main()


# "V:\NBS_Data\PBA_Alaska_UTM03N_Modeling"
# UTMN 03 through 07 folders exist
# /metadata/pba_alaska_utm03n_modeling
# same each utm has a table
# \\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBA_Alaska_UTM03N_Modeling
# v drive literal