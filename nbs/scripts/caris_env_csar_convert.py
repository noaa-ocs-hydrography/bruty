import pathlib
import sys
import os
import io


try:
    import caris.coverage as cc
    script_folder = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(script_folder, r'..\proc_io'))
    from fuse_dev.scripts.proc_io.csar_to_xyz import export_points
    from fuse_dev.scripts.proc_io.csar_raster_to_npy import csar_raster_to_npy
except ImportError as e:  # ModuleNotFoundError is not defined in 3.5
    print("Caris Python module not found -", str(e))


def convert_by_python(csar_path):
    data_type = cc.identify(csar_path)

    if data_type == cc.DatasetType.CLOUD:
        ret_path = npy_path = str(pathlib.Path(csar_path).with_suffix('.bruty.npz'))
        export_points(csar_path, npy_path)
        print("Made point clound npy")

    elif data_type == cc.DatasetType.RASTER:

        ret_path = elev_path = str(pathlib.Path(csar_path).with_suffix('.elev.raster.npy'))
        uncert_path = str(pathlib.Path(csar_path).with_suffix('.uncrt.raster.npy'))
        geotransform_path = str(pathlib.Path(csar_path).with_suffix('.geotransform.txt'))
        csar_raster_to_npy(csar_path, elev_path, uncert_path, geotransform_path)
        print("made raster npy")
    else:
        ret_path = None
        print("didn't recognize dataset type")

    print(ret_path)
    return ret_path


def main():
    convert_by_python(sys.argv[1])


if __name__ == '__main__':

    # default_config_name = "default.config"

    ## turn prints into logger messages
    # orig_print = print
    # def print(*args, **kywds):
    #     f = io.StringIO()
    #     ky = kywds.copy()
    #     ky['file'] = f
    #     orig_print(*args, **ky)  # build the string
    #     LOGGER.info(f.getvalue()[:-1])  # strip the newline at the end
    main()

