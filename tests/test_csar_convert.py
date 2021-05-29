# convert_csar_carisbatch
from nbs.scripts.convert_csar import convert_csar_python

def test_python_raster():
    path = r"V:\NBS_Data\PBC_Northeast_UTM19N_MLLW\USACE\eHydro_NewEngland_CENAE\Processed\MA_30_COH_20160509_AD_20_4m_interp.csar"
    path = r"C:\Git_Repos\Bruty\tests\MA_30_COH_20160509_AD_20_4m_interp.csar"
    metadata = {
        'to_horiz_frame': "NAD83",
        'to_horiz_type': 'spc',
        'to_horiz_key': '2001',
        'vert_uncert_fixed': 5.0,
        'vert_uncert_vari': 0.05,
    }
    new_path = convert_csar_python(path, metadata)

def test_python_points():
    # r"V:\NBS_Data\PBC_Northeast_UTM19N_MLLW\NOAA_NCEI_OCS\BAGs\Processed\D00246_MB_VR_MLLW.bruty.npy"
    path = r"V:\NBS_Data\PBC_Northeast_UTM19N_MLLW\USACE\eHydro_NewEngland_CENAE\Processed\MA_30_COH_20130522_CS_20.bruty.npy"
    path = r"C:\Git_Repos\Bruty\tests\MA_30_COH_20130522_CS_20.csar"
    metadata = {
        'to_horiz_frame': "NAD83",
        'to_horiz_type': 'spc',
        'to_horiz_key': '2001',
        'vert_uncert_fixed': 5.0,
        'vert_uncert_vari': 0.05,
    }

    new_path = convert_csar_python(path, metadata)

