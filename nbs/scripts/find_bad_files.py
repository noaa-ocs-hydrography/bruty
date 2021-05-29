import os
import glob
import pathlib
import datetime
import pickle
import traceback

try:
    import caris.coverage as cc
    def check_csar(fname, find_depth_interp=False):
        data_type = cc.identify(fname)

        if data_type == cc.DatasetType.CLOUD:
            csar = cc.Cloud(fname)
        elif data_type == cc.DatasetType.RASTER:
            csar = cc.Raster(fname)
        elif data_type == cc.DatasetType.VRS:
            csar = cc.VRS(fname)
        else:
            return 'UNKNOWN', None
        bands = [b.lower() for b in csar.band_info.keys()]
        # for bnd in ('depth', 'depth_pos', 'elevation interpolated', 'elevation', 'index', 'status', 'uncertainty', 'xyz'):
        #     try:
        #         bands.remove(bnd)
        #     except ValueError:
        #         pass
        # if bands:
        if find_depth_interp:
            if 'depth interpolated' in bands and 'elevation interpolated' not in bands:
                print(fname)
                print("  " + str(bands))
        return data_type.name, tuple([(bandname, band.direction.name) for bandname, band in csar.band_info.items()]), \
               {bandname: (band.minimum, band.maximum) for bandname, band in csar.band_info.items()}

    is_caris = True

except:
    from nbs_utils.points_utils import from_npz, to_npz
    import numpy
    is_caris = False

    print("Caris module not found")

    def check_time(dirpath, fname, fix=None):
        # set fix to a tolerance of seconds, if the csar is older but not more than that many seconds then revise its modified time
        csar = pathlib.Path(os.path.join(dirpath, fname))
        csar_time = csar.stat().st_mtime
        for exportname in (csar.with_name(csar.name + ".tif"), csar.with_suffix(".bruty.npz")):
            try:
                ex_time = exportname.stat().st_mtime
                if ex_time < csar_time:
                    # "f" strings don't work in python 3.5 which caris is in
                    # print(f"{exportname} was older than {fname}")
                    # print(f"{int(csar_time-ex_time)}s   {datetime.datetime.fromtimestamp(ex_time)} vs {datetime.datetime.fromtimestamp(csar_time)}")
                    print(exportname, " was older than ", fname)
                    print(int(csar_time-ex_time), "secs   ", datetime.datetime.fromtimestamp(ex_time), " vs ",datetime.datetime.fromtimestamp(csar_time))
                    if fix is not None:
                        if csar_time-ex_time < fix:
                            os.utime(csar, times=(csar.stat().st_atime, ex_time))  # use mtime from the exported file
                            print("revised the modified time of ", csar)
                        else:
                            pass
            except FileNotFoundError:
                pass

    def update_npz(filename):
        npz_data = numpy.load(filename)
        try:
            npz_data['minmax']
        except KeyError:
            del npz_data  # make sure the file closes since we are about to rewrite it
            try:
                wkt, data = from_npz(filename)
                to_npz(filename, wkt, data)
            except Exception as e:
                print('----------------------', filename, "failed")
                traceback.print_exc()
                print(filename, "failed  ------------------------")


info = {'CLOUD':{}, 'RASTER': {}, 'VRS': {}, 'UNKNOWN': {}, 'ERROR OPENING': {}}
upside_down = []
for data_type in ("NBS_Data_Unqualified", "NBS_Data_Sensitive", "NBS_Data_Qualified"):
    # for (dirpath, dirname, filenames) in os.walk(r"\\nos.noaa\OCS\HSD\Projects\NBS\\" + data_type): #\PBC_Northeast_UTM19N_MLLW"):  # \eHydro_NewEngland_CENAE\Manual"):  #r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data"):
    #         for fname in filenames:
    #             if fname.lower().endswith(".bruty.npz"):  # ".csar"):
    #                 full_path = os.path.join(dirpath, fname)
    for full_path in glob.glob(fr"\\nos.noaa\OCS\HSD\Projects\NBS\{data_type}\**\*.bruty.npz",recursive=True):
                    if is_caris:
                        if "manual" in dirpath.lower() and "jalbtcx" not in dirpath.lower():
                            try:
                                dtype, band_directions, band_minmax = check_csar(full_path)
                            except Exception as e:
                                info['ERROR OPENING'].setdefault('None', []).append(full_path)
                                # traceback.print_exc()
                                # print("  from", full_path)
                            else:
                                info[dtype].setdefault(band_directions, []).append(full_path)
                                try:
                                    min_elev, max_elev = band_minmax['Elevation']
                                    if max_elev > 0 and max_elev > abs(min_elev):
                                        print("Probably upside down, max is positive and larger magnitude than min (like +5, -1)")
                                        print(full_path)
                                        print(min_elev, max_elev)
                                        upside_down.append([min_elev, max_elev, full_path])
                                except KeyError:
                                    print("no elevation band", full_path)
                                else:
                                    try:
                                        min_xyz = band_minmax['XYZ'][0][-1]
                                        max_xyz = band_minmax['XYZ'][1][-1]
                                        if not(max_elev == max_xyz and min_elev == min_xyz):
                                            print("XYZ and Elevation min/max mismatch - are they in the same positive up direction")
                                            print(full_path)
                                            print(min_xyz, max_xyz)
                                            print(min_elev, max_elev)
                                    except KeyError:
                                        if dtype == 'CLOUD':
                                            print('XYZ not found in ', full_path)
                    else:
                        # check_time(dirpath, fname, 60)
                        update_npz(full_path)
if is_caris:
    for dtype, categories in info.items():
        print("\n--------------------------",dtype,"--------------------------")
        for cat, fnames in categories.items():
            print(cat, len(fnames))
    pickle.dump(info, open("info.pickle", "wb"))
    print("files that may be reversed (positive down)")
    for ud in upside_down:
        print(ud)
    print("finished")
