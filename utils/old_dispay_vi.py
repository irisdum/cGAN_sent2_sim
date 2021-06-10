import matplotlib.pyplot as plt

from constant.gee_constant import DICT_BAND_LABEL, DICT_BAND_X
from utils.display_image import plot_pre_post_pred
from utils.vi import compute_vi, diff_relative_metric, diff_metric


def display_one_image_vi(raster_array, fig, ax, vi, dict_band=None, title=None, cmap=None, vminmax=(0, 1),
                         path_csv=None, image_id=None):
    raster_vi = compute_vi(raster_array, vi, dict_band, path_csv=path_csv, image_id=image_id)

    if cmap is None:
        cmap = "RdYlGn"
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = vi
    im = ax.imshow(raster_vi, cmap=cmap, vmin=vminmax[0], vmax=vminmax[1])
    fig.colorbar(im, ax=ax, orientation="vertical")
    ax.set_title(title)
    if ax is None:
        plt.show()


def display_compare_vi(image_pre, image_post, vi, fig, ax, dict_band_pre, dict_band_post, figuresize=None,
                       vminmax=(0, 1), path_csv=None, image_id=None):
    if figuresize is None:
        figuresize = (20, 20)
    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=figuresize)
    display_one_image_vi(image_pre, fig, ax[0], vi, dict_band_pre, title="vi {} image pre".format(vi), vminmax=vminmax,
                         path_csv=path_csv, image_id=image_id)
    display_one_image_vi(image_post, fig, ax[1], vi, dict_band_post, title="vi {} image post".format(vi),
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    dr_vi = diff_relative_metric(image_pre, image_post, vi, dict_band_pre, dict_band_post, path_csv=path_csv,
                                 image_id=image_id)
    d_vi = diff_metric(image_pre, image_post, vi, dict_band_pre, dict_band_post, path_csv=path_csv, image_id=image_id)
    d_im = ax[2].imshow(d_vi, cmap="bwr", vmin=vminmax[0], vmax=vminmax[1])
    ax[2].set_title("differenced {}".format(vi))
    fig.colorbar(d_im, ax=ax[2], orientation="vertical")
    dr_im = ax[3].imshow(dr_vi, cmap="bwr", vmin=vminmax[0], vmax=vminmax[1])
    ax[3].set_title("relative differenced {}".format(vi))
    fig.colorbar(dr_im, ax=ax[3], orientation="vertical")
    plt.show()


def plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi, image_id=None, path_csv=None):
    plot_pre_post_pred(image_pre_fire, image_post_fire, image_pred)
    fig, ax = plt.subplots(1, 3, figsize=(40, 10))
    # vi_pre=compute_vi(image_pre_fire,vi)
    if path_csv is not None:
        vminmax = (0, 1)
    else:
        vminmax = (-1, 1)
    display_one_image_vi(image_pre_fire, fig, ax[0], vi, dict_band={"R": [4], "NIR": [7]}, title='Pre fire', cmap=None,
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    # vi_post=compute_vi(image_post,vi)
    display_one_image_vi(image_post_fire, fig, ax[1], vi, dict_band=None, title='GT post fire', cmap=None,
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    # vi_pred=compute_vi(image_pred,vi)
    display_one_image_vi(image_pred, fig, ax[2], vi, dict_band=None, title='Prediction post fire', cmap=None,
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    plt.show()


def compute_batch_vi(batch_x, batch_predict, batch_gt, max_im=100, vi="ndvi", liste_image_id=None, path_csv=None):
    """:param path_csv path to the csv file which contains min and max value"""
    n = batch_predict.shape[0]
    if n < max_im:
        max_im = n
    if liste_image_id is None:
        liste_image_id = [None for i in range(max_im)]
    for i in range(max_im):
        image_pre_fire = batch_x[i, :, :, :]
        image_post_fire = batch_gt[i, :, :, :]
        image_pred = batch_predict[i, :, :, ]
        print(image_pre_fire.shape, image_post_fire.shape, image_pred.shape)
        plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi, image_id=liste_image_id[i], path_csv=path_csv)
        gt_dvi = diff_metric(image_pre_fire, image_post_fire, vi, dict_band_pre={"R": [4], "NIR": [7]},
                             dict_band_post=DICT_BAND_LABEL, image_id=liste_image_id[i], path_csv=path_csv)
        pred_dvi = diff_metric(image_pre_fire, image_pred, vi, dict_band_pre=DICT_BAND_X,
                               dict_band_post=DICT_BAND_LABEL, image_id=liste_image_id[i], path_csv=path_csv)
        plot_compare_dvi(gt_dvi, pred_dvi)


def plot_compare_dvi(gt_dvi, pred_dvi):
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 30))
    display_one_image_vi(gt_dvi, fig2, ax2[0], "identity", dict_band=None, title='GT Relative difference', cmap="OrRd")
    display_one_image_vi(pred_dvi, fig2, ax2[1], "identity", dict_band=None, title='Pred Relative difference',
                         cmap="OrRd")
    plt.show()




def histo_val(dict_freq, ax=None, list_class=None, title=""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
        fig.suptitle(title)
    ax.bar(dict_freq.keys(), dict_freq.values(), tick_label=list_class, width=0.8)
    if list_class is not None:
        ax.set_xticks([i for i in range(0, len(list_class) + 1)], list_class)
        ax.set_xticklabels(list_class, rotation=70)
    # ax.set_xticks(dict_freq.keys())
    plt.show()


