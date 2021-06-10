# a python file where all the functions likned withe the preprocessing of the data just before the networks are implemented
# different methods are encode : normalization, centering or standardized values
import glob
import os
from pickle import dump, load
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from constant.gee_constant import DICT_BAND_X, DICT_BAND_LABEL, DICT_METHOD, DICT_TRANSLATE_BAND, \
    CONVERTOR
from constant.processing_constant import DICT_RESCALE, DICT_GROUP_BAND_LABEL, DICT_GROUP_BAND_X, S1_BANDS, S2_BANDS, \
    DICT_RESCALE_TYPE, FACT_STD_S2, FACT_STD_S1, DATA_RANGE
from utils.image_find_tbx import extract_tile_id, find_csv


def plot_one_band(raster_array, fig, ax, title="", cmap="bone"):
    """:param raster_array a numpy array
    Function that plot an np array with a colorbar"""
    # print("Imagse shape {}".format(raster_array))
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(raster_array, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation='vertical')
    if ax is None:
        plt.show()


def compute_image_stats(arrayX, arraylabel, dict_bandX=None, dictlabel=None, plot=False, stats="mean_std"):
    """Compute the statistics using the X array and its label. Statistics are computed for each band described in dict_band
    :param arrayX : a np array  with shape (n_image,m,m,n_channel)
    :param arraylabel : a np array  with shape (n_image,m,m,n_channel) wich correspond to the label array """
    assert stats in ["mean_std", "min_max"], "Stats function {} undefined".format(stats)
    if dict_bandX is None:
        dict_bandX = DICT_BAND_X
    if dictlabel is None:
        dictlabel = DICT_BAND_LABEL
    dict_stats = {}
    for band in dict_bandX:  # First go through all the image band defined in dictbandX
        n_images = len(dict_bandX[band])
        # if band in dictlabel:
        # n_images += len(dictlabel[band])
        if plot:
            fig, ax = plt.subplots(1, n_images, figsize=(20, 20))
        for i, b_index in enumerate(dict_bandX[band]):  # go through all the index given for this band in dataX
            if i == 0:
                band_array = arrayX[:, :, b_index]
                if plot:
                    plot_one_band(band_array, fig, ax[i])
            else:
                # print("we add another band")
                band_array = np.append(band_array, arrayX[:, :, b_index])
                if plot:
                    plot_one_band(arrayX[:, :, b_index], fig, ax[i],
                                  title="DATA X band {} index {}".format(band, b_index))
        if band in dictlabel:  # if this band is also in label
            # print("{} is also in label".format(band))
            for i, index in enumerate(dictlabel[band]):
                band_array = np.append(band_array, arraylabel[:, :, index])
                if plot:
                    plot_one_band(arraylabel[:, :, index], fig, ax[i + len(dict_bandX[band]) - 1],
                                  title="LABEL {}".format(band))
        if plot:
            plt.show()
        band_stat1, band_stat2 = compute_band_stats(band_array, stats)
        dict_stats.update({band: (band_stat1, band_stat2)})
    return dict_stats




def compute_band_stats(band_array, stats):
    if stats == "mean_std":
        stat1, stat2 = band_array.mean(), band_array.std()
    else:
        stat1, stat2 = band_array.min(), band_array.max()
    # print(stat1,stat2)
    return stat1, stat2


def positive_standardization(pixels, mean, std):
    """pixels an array
    mean a float
    std a float"""
    pixels = (pixels - mean) / std
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0
    return pixels


def normalization(pixels, min, max):
    pixels = (pixels - min) / (max - min)
    return pixels


def centering(pixels, mean, std):
    return pixels - mean


def rescaling_function(methode):
    # print("we use {}".format(methode))
    if methode == "normalization":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / (stat2 - stat1)
            return pixels
    elif methode == "normalization11":  # normalize between -1 and 1
        def method(pixels, stat1, stat2):
            pixels = 2 * (pixels - stat1) / (stat2 - stat1) - 1
            return pixels
    elif methode == "center_norm11":
        def method(pixels, stat1, stat2):
            val = (stat2 - stat1) / 2
            pixels = (pixels - val) / val
            return pixels
    elif methode == "center_norm11_r":
        def method(pixels, stat1, stat2):
            val = (stat2 - stat1) / 2
            pixels = pixels * val + val
            return pixels
    elif methode == "standardization11":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / stat2
            # pixels = np.clip(pixels, -1.0, 1.0)
            return pixels
    elif methode == "standardization":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / stat2
            pixels = np.clip(pixels, -1.0, 1.0)
            pixels = (pixels + 1.0) / 2.0
            return pixels
    elif methode == "normalization11_r":
        def method(pixels, stat1, stat2):
            pixels = (pixels + 1) / 2 * (stat2 - stat1) + stat1
            return pixels
    elif methode == "normalization_r":
        def method(pixels, stat1, stat2):
            pixels = pixels * (stat2 - stat1) + stat1
            return pixels
    elif methode == "centering_r":
        def method(pixels, stat1, stat2):
            return pixels + stat1
    else:
        def method(pixels, stat1, stat2):
            return pixels - stat1
    return method




def conv1D_dim(tuple_dim):
    return (tuple_dim[0] * tuple_dim[1] * tuple_dim[2] * tuple_dim[3], 1)


def rescale_array(batch_X: np.array, batch_label, dict_group_band_X=None, dict_group_band_label=None,
                  dict_rescale_type=None, s1_log=True, dict_scale=None, invert=False, s2_bands=S2_BANDS,
                  s1_bands=S1_BANDS, fact_scale2=FACT_STD_S2, fact_scale1=FACT_STD_S1, clip_s2=True) -> Tuple[
    np.array, np.array, dict]:
    """

    Args:
        clip_s2:
        fact_scale: float, the S2 bands will be multiplied by this factor after rescaling. Will only be appled to the
        bands defined in s2_bands
        batch_X: a numpy array
        batch_label: a numpy array , should have the same three first dimensions. Last dimension is the sprectral one
        dict_group_band_X: a dictionnary, gives indication on the index localisation of the band in the batch_X
        dict_group_band_label: a dictionnary, gives indication on the index localisation of the band in the batch_label
        dict_rescale_type: a dictionnary, for each group band defined in the input dictionnaires, gives the string
        method to use, the string should be associated in the method define in sklearn_scale
        s1_log : boolean, if set to True the sar bands are going to pe passed through 10*log10(x) function
    Returns:
        rescaled_batch_X : a numpy array, the rescaled batch_X
        ,rescaled_batch_label : a numpy array the rescaled batch_label
        dict_scale : a dictionnary, keys are string and values are the sklearn.processing Scaler. For each group band,
        gives it corresponding Scaler method.


    """
    dict_scaler = {}
    if dict_group_band_label is None:
        dict_group_band_label = DICT_GROUP_BAND_LABEL
    if dict_group_band_X is None:
        dict_group_band_X = DICT_GROUP_BAND_X
    if dict_rescale_type is None:
        dict_rescale_type = DICT_RESCALE_TYPE
    if dict_scale is None:
        dict_scale = {}
        for bands in s1_bands:
            dict_scale.update({bands: None})
        for bands in s2_bands:
            dict_scale.update({bands: None})

    rescaled_batch_X = np.zeros(batch_X.shape)
    rescaled_batch_label = np.zeros(batch_label.shape)
    # we deal with S1 normalization
    for group_bands in s1_bands:
        # all s1 band are in dict_band_X
        data_sar_band = batch_X[:, :, :, dict_group_band_X[group_bands]]
        nbands = len(dict_group_band_X[group_bands])
        if s1_log:
            data_nan_sar = np.copy(data_sar_band)
            data_nan_sar[data_nan_sar < 0] = float("nan")
            print(
                "Remove the negative values in order to have no error in the log : negative value will be replaced using"
                "knn algorithm")
            data_sar_band = replace_batch_nan_knn(data_nan_sar, [i for i in range(nbands)])
            data_sar_band = 10 * np.log10(data_sar_band)
        init_shape = data_sar_band.shape
        data_flatten_sar_band = data_sar_band.reshape(
            conv1D_dim(data_sar_band.shape))  # Modify into 2D array as required for sklearn
        output_data, sar_scale = sklearn_scale(dict_rescale_type[group_bands], data_flatten_sar_band,
                                               scaler=dict_scale[group_bands], fact_scale=fact_scale1, invert=invert)
        rescaled_batch_X[:, :, :, dict_group_band_X[group_bands]] = output_data.reshape(init_shape)  # reshape it
        dict_scaler.update({group_bands: sar_scale})
    for group_bands in s2_bands:
        m = batch_X.shape[0]  # the nber of element in batch_X
        data = np.concatenate((batch_X[:, :, :, dict_group_band_X[group_bands]],
                               batch_label[:, :, :, dict_group_band_label[group_bands]]))
        global_shape = data.shape
        data_flatten = data.reshape(conv1D_dim(data.shape))

        flat_rescale_data, scale_s2 = sklearn_scale(dict_rescale_type[group_bands], data_flatten,
                                                    scaler=dict_scale[group_bands], invert=invert,
                                                    fact_scale=fact_scale2)
        if clip_s2:  # we clip between -1 and 1
            flat_rescale_data = np.clip(flat_rescale_data, DATA_RANGE[0], DATA_RANGE[1])

        rescale_global_data = flat_rescale_data.reshape(global_shape)
        # print("rescale_global_shape {} sub {} fit in {} & label {}".format(rescale_global_data.shape,
        #                                                         rescale_global_data[:m , :, :, :].shape,
        #                                                         rescaled_batch_X[:, :, :, dict_group_band_X[group_bands]].shape,rescaled_batch_label.shape))
        rescaled_batch_X[:, :, :, dict_group_band_X[group_bands]] = rescale_global_data[:m, :, :, :]
        rescaled_batch_label[:, :, :, dict_group_band_label[group_bands]] = rescale_global_data[m:, :, :, :]
        dict_scaler.update({group_bands: scale_s2})

    return rescaled_batch_X, rescaled_batch_label, dict_scaler


def sklearn_scale(scaling_method, data, scaler=None, invert=False, fact_scale=1):
    """
    Args:
        scaling_method: string, name of the method currently only StandardScaler works
        data: input data array to be rescaled
        scaler : a sklearn Scaler
    Returns:
        data_rescale : the rescaled input numpy array (data)
        scaler :  the sklearn.processing method used
    """
    assert scaling_method in ["StandardScaler"], "The method name is not defined {}".format(scaling_method)
    if scaling_method == "StandardScaler":
        if scaler is None:
            print("No scaler was defined before")
            scaler = StandardScaler()
            scaler.fit(data)
        else:
            if invert:
                return scaler.inverse_transform(data * 1 / fact_scale), scaler
        data_rescale = scaler.transform(data)
        return data_rescale * fact_scale, scaler
    else:
        return data, None


def knn_model(data):
    knn = KNNImputer(n_neighbors=5)
    return knn.fit_transform(data)


def replace_batch_nan_knn(batch, lband_index):
    print("Important the index of the bands in lband_index should be index that follow each other")
    knn_batch = np.copy(batch)
    for b in lband_index:
        with parallel_backend("loky", inner_max_num_threads=2):
            list_arr_band = Parallel(n_jobs=2)(delayed(knn_model)(data) for data in batch[:, :, :, b])
        knn_batch[:, :, :, b] = np.array(list_arr_band)
    return knn_batch


def save_all_scaler(scaler_dict: dict, path_dir: str):
    """

    Args:
        scaler_dict: dict where keys corresponds to a group of band,
        path_dir: d

    Returns:

    """
    for bands in scaler_dict:
        save_model(scaler_dict[bands], path_dir + bands + "_scaler.pkl")


def save_model(scaler: StandardScaler, path: str):
    """

    Args:
        scaler:
        scaler_name: name used to save the model

    Returns:

    """
    dump(scaler, open(path, 'wb'))


def load_scaler(input_dir, bands) -> dict:
    """

    Args:
        input_dir: path to directory where pkl scaler model are saved
        bands: string group bands name (used to save the model)

    Returns:

    """
    l = glob.glob("{}*{}*.pkl".format(input_dir, bands))
    assert len(l) == 1, "Found more or less pkl model for group {} we found {}".format(
        "{}*{}*.pkl".format(input_dir, bands), l)
    path_scaler = l[0]
    # load the scaler
    scaler = load(open(path_scaler, 'rb'))
    return {bands: scaler}

def load_dict_scaler(input_dir: str, l_band_group: list) -> dict:
    """

    Args:
        input_dir:
        l_band_group:

    Returns:

    """
    dict_scaler = {}
    for bands in l_band_group:
        dict_scaler.update(load_scaler(input_dir, bands))
    return dict_scaler
