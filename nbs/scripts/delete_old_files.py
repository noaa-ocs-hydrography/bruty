import os, glob

files = [f for f in glob.glob(p+r"V:\bruty_tile_exports\PBB17\**\Tile*20221001*", recursive=True) if "Public" not in f]
for f in files:
    try:
        os.remove(f)
    except:
        pass