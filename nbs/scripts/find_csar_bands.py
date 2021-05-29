import os
import pathlib
import datetime

try:
    import caris.coverage as cc
    def check_csar(fname):
        data_type = cc.identify(fname)

        if data_type == cc.DatasetType.CLOUD:
            csar = cc.Cloud(fname)
        elif data_type == cc.DatasetType.RASTER:
            csar = cc.Raster(fname)
        else:
            return
        bands = [b.lower() for b in csar.band_info.keys()]
        # for bnd in ('depth', 'depth_pos', 'elevation interpolated', 'elevation', 'index', 'status', 'uncertainty', 'xyz'):
        #     try:
        #         bands.remove(bnd)
        #     except ValueError:
        #         pass
        # if bands:
        if 'depth interpolated' in bands and 'elevation interpolated' not in bands:
            print(fname)
            print("  " + str(bands))

except:
    print("Caris module not found")

def check_time(dirpath, fname, fix=None):
    # set fix to a tolerance of seconds, if the csar is older but not more than that many seconds then revise its modified time
    csar = pathlib.Path(os.path.join(dirpath, fname))
    csar_time = csar.stat().st_mtime
    for exportname in (csar.with_name(csar.name + ".tif"), csar.with_suffix(".bruty.npz")):
        try:
            ex_time = exportname.stat().st_mtime
            if ex_time < csar_time:
                print(f"{exportname} was older than {fname}")
                print(f"{int(csar_time-ex_time)}s   {datetime.datetime.fromtimestamp(ex_time)} vs {datetime.datetime.fromtimestamp(csar_time)}")
                if fix is not None:
                    if csar_time-ex_time < fix:
                        os.utime(csar, times=(csar.stat().st_atime, ex_time))  # use mtime from the exported file
                        print(f"revised the modified time of {csar}")
                    else:
                        pass
        except FileNotFoundError:
            pass


for (dirpath, dirname, filenames) in os.walk(r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data"):  # \PBC_Northeast_UTM19N_MLLW\USACE\eHydro_NewEngland_CENAE\Manual"):  #r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data"):
        for fname in filenames:
            if fname.lower().endswith(".csar"):
                if "manual" in dirpath.lower():
                    # check_csar(os.path.join(dirpath, fname))
                    pass
                check_time(dirpath, fname, 60)

