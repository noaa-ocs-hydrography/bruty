import os
import glob
import pathlib

def compress(root):
    rootpath = pathlib.Path(root)
    os.makedirs(rootpath.parent.joinpath("zips"), exist_ok=True)
    for dirname in glob.glob(str(rootpath)):
        path = pathlib.Path(dirname)
        outfile = path.parent.joinpath("zips", path.name).with_suffix(".zip")
        if not os.path.exists(outfile) and path.name != "zips":
            command = fr'"c:\Program Files\7-Zip\7z.exe" a {outfile} {path}'
            os.system(command)


def decompress(zips, dest):
    """ Skips existing files
    Parameters
    ----------
    zips
    dest

    Returns
    -------

    """
    os.makedirs(dest, exist_ok=True)
    rootpath = pathlib.Path(zips)
    for zipname in glob.glob(str(rootpath)):
        command = fr'"c:\Program Files\7-Zip\7z.exe" x -aos -o{dest} {zipname}'
        os.system(command)


compressing = False
if compressing:
    compress(r"E:\bruty_databases\PBC_Northeast_utm19n_MLLW_Tile32*")
    compress(r"E:\bruty_databases\PBC_Northeast_utm19n_MLLW_Tile5_*")
    compress(r"E:\bruty_databases\*")
else:
    decompress(r"I:\bruty_databases\*.zip", r"d:\bruty_databases")
