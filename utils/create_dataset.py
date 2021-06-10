import os

from osgeo import gdal

from constant.model_constant import TRAINING_DIR
from constant.storing_constant import XDIR, LABEL_DIR
from utils.image_find_tbx import create_safe_directory


def tiff_2_array(path_tif: str):
    assert os
    raster = gdal.Open(path_tif)
    return raster.ReadAsArray()





def make_dataset_hierarchy(path_dataset: str):
    assert path_dataset[-1] == "/", "Wrong path should end with / not {}".format(path_dataset)
    create_safe_directory(path_dataset)
    for sub_dir in TRAINING_DIR:
        os.mkdir(path_dataset + sub_dir)
        os.mkdir(path_dataset + sub_dir + XDIR)
        os.mkdir(path_dataset + sub_dir + LABEL_DIR)