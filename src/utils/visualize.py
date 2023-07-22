import os
import numpy as np
from skimage.transform import rescale, resize
# from data.IO import *
# from data.utils import get_bounds, make_binary, pad_data, mkdir
from scipy.ndimage.morphology import distance_transform_edt
import imageio

__all__ = ['generate_mip', 'normalize_to_grayscale', 'overlay_mask_to_image', 'make_sshot', 'generate_sshot_overview',
           'organ_cmap', 'diff_cmap', 'generate_sshot_lite', 'generate_mip_overview',
           'make_sshot_contour', 'generate_video_one', 'generate_video_two', 'generate_video_two_in_one',
           'generate_video_mr_one', 'generate_video_mr_two', 'generate_video_mr_two_in_one']


diff_cmap = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
], dtype='float32')
diff_cmap = diff_cmap[:, [0, 2, 1]]

organ_cmap = np.array([
    [0, 0, 0],
    [50, 50, 255],
    [255, 50, 50],
    [192, 160, 255],
    [255, 255, 50],
    [100, 100, 255],
    [255, 120, 140],
    [191, 165, 136],
    [0, 140, 100],
    [120, 110, 60],
    [88, 201, 182],
    [230, 50, 230],
    [50, 50, 55],
    [100, 255, 100],
], dtype='float32')
organ_cmap = organ_cmap[:, [0, 2, 1]]

# to be continued...


def generate_cuts_mip(ref_masked_image, cut_metrics, ax, win_center=1374, win_width=750, data_res=None, unit=512):
    vessel_tree_mask = cut_metrics["estimated_reference_vessel_tree"]
    full_cuts_mask = make_binary(cut_metrics["full_cuts_severe_label_image"])
    partial_cuts_mask = make_binary(cut_metrics["partial_cuts_severe_label_image"])
    canvas = generate_mip(ref_masked_image, ax=ax, data_res=data_res)
    vessel_tree_mip = generate_mip(vessel_tree_mask, ax=ax, win_center=None, win_width=None, data_res=data_res)[:,:,0]
    full_cuts_mip = generate_mip(full_cuts_mask, ax=ax, win_center=None, win_width=None, data_res=data_res)[:,:,0]
    partial_cuts_mip = generate_mip(partial_cuts_mask, ax=ax, win_center=None, win_width=None, data_res=data_res)[:,:,0]
    inv_vessel_tree_mip = 1 - vessel_tree_mip
    inv_full_cuts_mip = 1 - full_cuts_mip
    inv_partial_cuts_mip = 1 - partial_cuts_mip
    canvas[:,:,0] = canvas[:,:,0] * inv_vessel_tree_mip
    canvas[:,:,2] = canvas[:,:,2] * inv_vessel_tree_mip
    canvas[:,:,0] = np.maximum(canvas[:,:,0], np.maximum(full_cuts_mip, partial_cuts_mip))
    canvas[:,:,1] = np.maximum(canvas[:,:,1], partial_cuts_mip) * inv_full_cuts_mip
    canvas[:,:,2] = canvas[:,:,2] * inv_full_cuts_mip * inv_partial_cuts_mip
    return canvas


def generate_mip(img, ax, win_center=1374, win_width=750, data_res=None, unit=512):
    # assuming img in order of z, y, x, while data_res is the order of in x, y, z

    ax = 2 if ax > 2 else ax
    ax = 0 if ax < 0 else ax

    data_res = [1.0, 1.0, 1.0] if data_res is None else data_res
    if len(data_res) != 3:
        raise ValueError('expect 3 for length of data_res, got {}.'.format(len(data_res)))

    if ax == 0:  # axial
        out_shape = [img.shape[1] * data_res[1], img.shape[2] * data_res[0]]
    elif ax == 1:  # coronal
        out_shape = [img.shape[0] * data_res[2], img.shape[2] * data_res[0]]
    else:  # sagittal
        out_shape = [img.shape[0] * data_res[2], img.shape[1] * data_res[1]]

    out_shape = [unit, int(unit * out_shape[1] / out_shape[0])] if out_shape[0] > out_shape[1] \
        else [int(unit * out_shape[0] / out_shape[1]), unit]

    mip_img = normalize_to_grayscale(np.max(img, axis=ax), win_center, win_width).astype(np.uint8)
    mip_img = np.flipud(mip_img) if ax != 0 else mip_img
    return resize(np.tile(mip_img[..., np.newaxis], (1, 1, 3)), out_shape, mode='constant')


def normalize_to_grayscale(image, win_center=None, win_width=None):
    image = image.astype(np.float32)
    if not win_center or not win_width:
        return (image - image.min()) / (image.max() - image.min() + np.finfo(float).eps) * 255
    else:
        hw = int(win_width / 2)
        image -= (win_center - hw)
        image[image < 0] = 0
        image /= win_width
        image[image > 1] = 1
        return image * 255


def overlay_mask_to_image(image, mask, cm, k, skip_bg=True):
    img_r = np.copy(image)
    img_g = np.copy(image)
    img_b = np.copy(image)

    mask[mask == 255] = len(cm) - 1
    lbs = np.unique(mask)

    for a in lbs:
        if skip_bg and a == 0:
            continue
        img_r[(mask == a)] = k * img_r[(mask == a)] + (1 - k) * cm[a, 0]
        img_g[(mask == a)] = k * img_g[(mask == a)] + (1 - k) * cm[a, 1]
        img_b[(mask == a)] = k * img_b[(mask == a)] + (1 - k) * cm[a, 2]
    img_rgb = np.concatenate((img_r[..., np.newaxis], img_b[..., np.newaxis],
                              img_g[..., np.newaxis]), axis=2).astype(np.uint8)
    return img_rgb


def make_sshot(image, mask, out_shape, win_center, win_width, cm, k):
    image = normalize_to_grayscale(image, win_center, win_width)
    oim = overlay_mask_to_image(image, mask, cm, k)
    return resize(oim, out_shape, mode='constant')


def make_sshot_contour(image, mask, out_shape, win_center, win_width, cm, k, r=2):
    labels = np.unique(mask)
    for l in labels:
        if l == 0:
            continue
        label_mask = (mask == l)
        dist_map = distance_transform_edt(label_mask)
        mask[(dist_map > r) * (label_mask > 0)] = 0
    return make_sshot(image, mask, out_shape, win_center, win_width, cm, k)


# sshot for three view at the central slice of the mask
# however without zooming in ROI, all three views rescaled to a fixed size
def generate_sshot_lite(image, mask, win_center=None, win_width=None, cm=organ_cmap, unit=256, k=0.5):
    if np.sum(mask) == 0:
        return None

    zmin, zmax, ymin, ymax, xmin, xmax = get_bounds(mask > 0)
    zc, yc, xc = int((zmin + zmax) / 2), int((ymin + ymax) / 2), int((xmin + xmax) / 2)
    out_shape = [unit, unit]
    canvas = np.zeros((unit, 3 * unit, 3), dtype='uint8')

    # axial
    img_slice = image[zc, :, :]
    mask_slice = mask[zc, :, :]
    out_slice = make_sshot(img_slice, mask_slice, out_shape, win_center, win_width, cm, k)
    canvas[:, :unit, :] = out_slice * 255.0

    # coronal
    img_slice = image[::-1, yc, :]
    mask_slice = mask[::-1, yc, :]
    out_slice = make_sshot(img_slice, mask_slice, out_shape, win_center, win_width, cm, k)
    canvas[:, unit: 2 * unit, :] = out_slice * 255.0

    # sagittal
    img_slice = image[::-1, :, xc]
    mask_slice = mask[::-1, :, xc]
    out_slice = make_sshot(img_slice, mask_slice, out_shape, win_center, win_width, cm, k)
    canvas[:, 2 * unit:, :] = out_slice * 255.0

    return canvas


def generate_mip_overview(image, tr_mask, pd_mask, specs, ax=1, win_center=1374, win_width=750, unit=512):
    tr_mip = generate_mip(image * (tr_mask == 0), ax=ax,
                          win_center=win_center, win_width=win_width, data_res=specs[1], unit=unit)
    pd_mip = generate_mip(image * (pd_mask == 0), ax=ax,
                          win_center=win_center, win_width=win_width, data_res=specs[1], unit=unit)
    return np.concatenate((tr_mip, pd_mip), axis=1)


def get_bounds(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return xmin, xmax, ymin, ymax, zmin, zmax


def generate_video_one(img_data, gt_mask_data, voi, img_specs, sub, out_path, ffmpeg_path,
                      win_center=1024, win_width=500, bf=5, unit=256, alpha=0.5, cm=organ_cmap,
                      single_label=None, mode='axial', fps=5, show_contour=False):

    if mode not in ['axial', 'coronal', 'sagittal']:
        mode = 'axial'
    out_pic_path = os.path.join(out_path, mode)
    mkdir(out_pic_path)

    data_dim, data_res = img_specs[0], img_specs[1]
    if single_label is not None:
        gt_mask_data = np.array((gt_mask_data == single_label), dtype=np.uint8) * 255
    if np.sum(gt_mask_data) == 0:
        print('nothing in the gt mask, skip!')
        return

    # get bbox
    zmin, zmax, ymin, ymax, xmin, xmax = get_bounds(voi)
    # get the range with buffer
    zlb, zub = max(0, zmin - bf), min(data_dim[2], zmax + 1 + bf)
    ylb, yub = max(0, ymin - bf), min(data_dim[1], ymax + 1 + bf)
    xlb, xub = max(0, xmin - bf), min(data_dim[0], xmax + 1 + bf)
    # get the shape with realistic aspect ratio
    if mode == 'axial':
        out_shape = [(yub - ylb) * data_res[1], (xub - xlb) * data_res[0]]
        kmin, kmax = max(0, zmin - bf), min(data_dim[2], zmax + 1 + bf)
        jmin, jmax = ylb, yub
        imin, imax = xlb, xub
    elif mode == 'coronal':
        out_shape = [(zub - zlb) * data_res[2], (xub - xlb) * data_res[0]]
        kmin, kmax = max(0, ymin - bf), min(data_dim[1], ymax + 1 + bf)
        img_data = np.transpose(img_data, (1, 0, 2))
        gt_mask_data = np.transpose(gt_mask_data, (1, 0, 2))
        jmin, jmax = zlb, zub
        imin, imax = xlb, xub
    else:  # sagittal
        out_shape = [(zub - zlb) * data_res[2], (yub - ylb) * data_res[1]]
        kmin, kmax = max(0, xmin - bf), min(data_dim[0], xmax + 1 + bf)
        img_data = np.transpose(img_data, (2, 0, 1))
        gt_mask_data = np.transpose(gt_mask_data, (2, 0, 1))
        jmin, jmax = zlb, zub
        imin, imax = ylb, yub

    # fit the shape with the unit size
    if out_shape[0] > out_shape[1]:
        out_shape = [unit, int(unit * out_shape[1] / out_shape[0])]
        pad_l = np.array([0, int((unit - out_shape[1]) / 2), 0])
        pad_u = np.array([0, unit - out_shape[1] - pad_l[1], 0])

    else:
        out_shape = [int(unit * out_shape[0] / out_shape[1]), unit]
        pad_l = np.array([int((unit - out_shape[0]) / 2), 0, 0])
        pad_u = np.array([unit - out_shape[0] - pad_l[0], 0, 0])

    for k in range(kmin, kmax):
        if mode == 'axial':
            img_slice = img_data[k, jmin:jmax, imin:imax]
            gt_mask_slice = gt_mask_data[k, jmin:jmax, imin:imax]
        else:
            img_slice = img_data[k, jmax:jmin:-1, imin:imax]
            gt_mask_slice = gt_mask_data[k, jmax:jmin:-1, imin:imax]

        # get two slice-wise sshots, 1: reference, 2: gt
        out_slice_1 = make_sshot(img_slice, gt_mask_slice, out_shape, win_center, win_width, cm, 1)
        if show_contour:
            out_slice_2 = make_sshot_contour(img_slice, gt_mask_slice, out_shape, win_center, win_width, cm, alpha)
        else:
            out_slice_2 = make_sshot(img_slice, gt_mask_slice, out_shape, win_center, win_width, cm, alpha)

        # make the size as unit x unit
        out_slice_1 = pad_data(out_slice_1, pad_l, pad_u, 'constant')
        out_slice_2 = pad_data(out_slice_2, pad_l, pad_u, 'constant')
        # combine
        out_slice = np.concatenate((out_slice_1, out_slice_2), axis=1)
        out_file = os.path.join(out_pic_path, sub + '_%s_%04d.png' % (mode, (k + 1)))
        imageio.imwrite(out_file, out_slice)

    # generate mp4 out of the png frames
    filename_format = out_pic_path + os.path.sep + sub + '_' + mode + '_%04d.png'
    out_mp4_file = os.path.join(out_path, sub + '_%s.mp4' % mode)
    if os.path.isfile(out_mp4_file):
        os.remove(out_mp4_file)  # remove the mp4 if already exists
    cmd = '%s -r %d -f image2 -s 1920x1080 ' \
          '-start_number %d -i %s -vframes %d -vcodec libx264 -crf 25 -pix_fmt yuv420p %s' \
          % (ffmpeg_path, fps, kmin, filename_format, (kmax - kmin + 1), out_mp4_file)
    os.system(cmd)


def generate_video_two(img_data, gt_mask_data, pred_mask_data, voi, img_specs, sub, out_path, ffmpeg_path,
                      win_center=1024, win_width=500, bf=5, unit=256, alpha=0.5, cm=organ_cmap,
                      single_label=None, mode='axial', fps=5, show_contour=False):
    # '/cm/shared/apps/ffmpeg/4.3/bin/ffmpeg'
    if mode not in ['axial', 'coronal', 'sagittal']:
        mode = 'axial'
    out_pic_path = os.path.join(out_path, mode)
    mkdir(out_pic_path)

    data_dim, data_res = img_specs[0], img_specs[1]
    if single_label is not None:
        gt_mask_data = np.array((gt_mask_data == single_label), dtype=np.uint8) * 255
        pred_mask_data = np.array((pred_mask_data == single_label), dtype=np.uint8) * 255
    if np.sum(gt_mask_data) == 0:
        print('nothing in the gt mask, skip!')
        return

    # get bbox
    zmin, zmax, ymin, ymax, xmin, xmax = get_bounds(voi)
    # get the range with buffer
    zlb, zub = max(0, zmin - bf), min(data_dim[2], zmax + 1 + bf)
    ylb, yub = max(0, ymin - bf), min(data_dim[1], ymax + 1 + bf)
    xlb, xub = max(0, xmin - bf), min(data_dim[0], xmax + 1 + bf)
    # get the shape with realistic aspect ratio
    if mode == 'axial':
        out_shape = [(yub - ylb) * data_res[1], (xub - xlb) * data_res[0]]
        kmin, kmax = max(0, zmin - bf), min(data_dim[2], zmax + 1 + bf)
        jmin, jmax = ylb, yub
        imin, imax = xlb, xub
    elif mode == 'coronal':
        out_shape = [(zub - zlb) * data_res[2], (xub - xlb) * data_res[0]]
        kmin, kmax = max(0, ymin - bf), min(data_dim[1], ymax + 1 + bf)
        img_data = np.transpose(img_data, (1, 0, 2))
        gt_mask_data = np.transpose(gt_mask_data, (1, 0, 2))
        pred_mask_data = np.transpose(pred_mask_data, (1, 0, 2))
        jmin, jmax = zlb, zub
        imin, imax = xlb, xub
    else:  # sagittal
        out_shape = [(zub - zlb) * data_res[2], (yub - ylb) * data_res[1]]
        kmin, kmax = max(0, xmin - bf), min(data_dim[0], xmax + 1 + bf)
        img_data = np.transpose(img_data, (2, 0, 1))
        gt_mask_data = np.transpose(gt_mask_data, (2, 0, 1))
        pred_mask_data = np.transpose(pred_mask_data, (2, 0, 1))
        jmin, jmax = zlb, zub
        imin, imax = ylb, yub

    # fit the shape with the unit size
    if out_shape[0] > out_shape[1]:
        out_shape = [unit, int(unit * out_shape[1] / out_shape[0])]
        pad_l = np.array([0, int((unit - out_shape[1]) / 2), 0])
        pad_u = np.array([0, unit - out_shape[1] - pad_l[1], 0])

    else:
        out_shape = [int(unit * out_shape[0] / out_shape[1]), unit]
        pad_l = np.array([int((unit - out_shape[0]) / 2), 0, 0])
        pad_u = np.array([unit - out_shape[0] - pad_l[0], 0, 0])

    for k in range(kmin, kmax):
        if mode == 'axial':
            img_slice = img_data[k, jmin:jmax, imin:imax]
            gt_mask_slice = gt_mask_data[k, jmin:jmax, imin:imax]
            pred_mask_slice = pred_mask_data[k, jmin:jmax, imin:imax]
        else:
            img_slice = img_data[k, jmax:jmin:-1, imin:imax]
            gt_mask_slice = gt_mask_data[k, jmax:jmin:-1, imin:imax]
            pred_mask_slice = pred_mask_data[k, jmax:jmin:-1, imin:imax]

        # get three slice-wise sshots, 1: reference, 2: gt, 3: pred

        out_slice_1 = make_sshot(img_slice, gt_mask_slice, out_shape, win_center, win_width, cm, 1)
        if show_contour:
            out_slice_2 = make_sshot_contour(img_slice, gt_mask_slice, out_shape, win_center, win_width, cm, alpha)
            out_slice_3 = make_sshot_contour(img_slice, pred_mask_slice, out_shape, win_center, win_width, cm, alpha)
        else:
            out_slice_2 = make_sshot(img_slice, gt_mask_slice, out_shape, win_center, win_width, cm, alpha)
            out_slice_3 = make_sshot(img_slice, pred_mask_slice, out_shape, win_center, win_width, cm, alpha)

        # make the size as unit x unit
        out_slice_1 = pad_data(out_slice_1, pad_l, pad_u, 'constant')
        out_slice_2 = pad_data(out_slice_2, pad_l, pad_u, 'constant')
        out_slice_3 = pad_data(out_slice_3, pad_l, pad_u, 'constant')
        # combine
        out_slice = np.concatenate((out_slice_1, out_slice_2, out_slice_3), axis=1)
        out_file = os.path.join(out_pic_path, sub + '_%s_%04d.png' % (mode, (k + 1)))
        imageio.imwrite(out_file, out_slice)

    # generate mp4 out of the png frames
    filename_format = out_pic_path + os.path.sep + sub + '_' + mode + '_%04d.png'
    out_mp4_file = os.path.join(out_path, sub + '_%s.mp4' % mode)
    if os.path.isfile(out_mp4_file):
        os.remove(out_mp4_file)  # remove the mp4 if already exists
    cmd = '%s -r %d -f image2 -s 1920x1080 ' \
          '-start_number %d -i %s -vframes %d -vcodec libx264 -crf 25 -pix_fmt yuv420p %s' \
          % (ffmpeg_path, fps, kmin, filename_format, (kmax - kmin + 1), out_mp4_file)
    os.system(cmd)


# only applies to binary segmentation
def generate_video_two_in_one(img_data, gt_mask_data, pred_mask_data, voi, img_specs, sub, out_path, ffmpeg_path,
                      win_center=1024, win_width=500, bf=5, unit=256, alpha=0.5, cm=organ_cmap,
                      mode='axial', fps=5, show_contour=False):

    generate_video_one(img_data,
                       np.array((2 * (gt_mask_data > 0) + (pred_mask_data > 0)), dtype='uint8'),
                       voi,
                       img_specs, sub, out_path, ffmpeg_path,
                       win_center=win_center, win_width=win_width, bf=bf, unit=unit, alpha=alpha, cm=cm,
                       single_label=None, mode=mode, fps=fps, show_contour=show_contour)


def generate_video_mr_one(img_data, gt_mask_data, voi, img_specs, sub, out_path, ffmpeg_path,
                      norm_top_percentile=99.5, norm_bottom_percentile=0.05, bf=5, unit=256, alpha=0.5, cm=organ_cmap,
                      single_label=None, mode='axial', fps=5, show_contour=False):

    v_top = np.percentile(img_data, norm_top_percentile)
    v_bottom = np.percentile(img_data, norm_bottom_percentile)
    img_data = np.clip((img_data - v_bottom) / (v_top - v_bottom), 0.0, 1.0)
    generate_video_one(img_data, gt_mask_data, voi,
                       img_specs, sub, out_path, ffmpeg_path,
                       win_center=None, win_width=None, bf=bf, unit=unit, alpha=alpha, cm=cm,
                       single_label=single_label, mode=mode, fps=fps, show_contour=show_contour)


def generate_video_mr_two(img_data, gt_mask_data, pred_mask_data, voi, img_specs, sub, out_path, ffmpeg_path,
                      norm_top_percentile=99.5, norm_bottom_percentile=0.05, bf=5, unit=256, alpha=0.5, cm=organ_cmap,
                      single_label=None, mode='axial', fps=5, show_contour=False):

    v_top = np.percentile(img_data, norm_top_percentile)
    v_bottom = np.percentile(img_data, norm_bottom_percentile)
    img_data = np.clip((img_data - v_bottom) / (v_top - v_bottom), 0.0, 1.0)
    generate_video_two(img_data, gt_mask_data, pred_mask_data, voi,
                       img_specs, sub, out_path, ffmpeg_path,
                       win_center=None, win_width=None, bf=bf, unit=unit, alpha=alpha, cm=cm,
                       single_label=single_label, mode=mode, fps=fps, show_contour=show_contour)


def generate_video_mr_two_in_one(img_data, gt_mask_data, pred_mask_data, voi, img_specs, sub, out_path, ffmpeg_path,
                      norm_top_percentile=99.5, norm_bottom_percentile=0.05, bf=5, unit=256, alpha=0.5, cm=organ_cmap,
                      mode='axial', fps=5, show_contour=False):

    v_top = np.percentile(img_data, norm_top_percentile)
    v_bottom = np.percentile(img_data, norm_bottom_percentile)
    img_data = np.clip((img_data - v_bottom) / (v_top - v_bottom), 0.0, 1.0)
    generate_video_two_in_one(img_data, gt_mask_data, pred_mask_data, voi,
                       img_specs, sub, out_path, ffmpeg_path,
                       win_center=None, win_width=None, bf=bf, unit=unit, alpha=alpha, cm=cm,
                       mode=mode, fps=fps, show_contour=show_contour)


def generate_sshot_overview(image, tr_mask, pd_mask, specs,
                            win_center=None, win_width=None,
                            cm=diff_cmap, unit=128, k=0.5, bf=20, mode=7):

    mode = 7 if mode > 7 else mode
    mode = 0 if mode < 0 else mode
    if mode == 0:
        return None
    b_axi = (mode % 2 > 0)
    b_cor = (mode % 4 > 1)
    b_sag = (mode > 4)

    labels = list(np.unique(tr_mask))
    if 0 in labels:
        labels.remove(0)
    if not labels:
        return None

    num_cols = int(b_axi + b_cor + b_sag)
    num_rows = len(labels)
    canvas = np.zeros((num_rows * unit, num_cols * unit, 3), dtype='uint8')
    data_dim, data_res = specs[0], specs[1]

    for i, l in enumerate(labels):
        zmin, zmax, ymin, ymax, xmin, xmax = get_bounds(tr_mask == l)
        # green: truth, red: prediction, yellow: overlapped region
        mask = np.array((2 * (tr_mask == l) + (pd_mask == l)), dtype='uint8')

        zc, yc, xc = int((zmin + zmax) / 2), int((ymin + ymax) / 2), int((xmin + xmax) / 2)
        zlb, zub = max(0, zmin - bf), min(data_dim[2], zmax + 1 + bf)
        ylb, yub = max(0, ymin - bf), min(data_dim[1], ymax + 1 + bf)
        xlb, xub = max(0, xmin - bf), min(data_dim[0], xmax + 1 + bf)

        j = 0
        if b_axi:
            img_slice = image[zc, ylb:yub, xlb:xub]
            mask_slice = mask[zc, ylb:yub, xlb:xub]
            out_shape = [(yub - ylb) * data_res[1], (xub - xlb) * data_res[0]]
            if out_shape[0] > out_shape[1]:
                out_shape = [unit, int(unit * out_shape[1] / out_shape[0])]
            else:
                out_shape = [int(unit * out_shape[0] / out_shape[1]), unit]
            out_slice = make_sshot(img_slice, mask_slice, out_shape, win_center, win_width, cm, k)
            # note that after make_sshot, the image value range from 0 to 1 as float
            canvas[i * unit: i * unit + out_shape[0], j * unit: j * unit + out_shape[1], :] = out_slice * 255.0
            j += 1

        if b_cor:
            img_slice = image[zub:zlb:-1, yc, xlb:xub]
            mask_slice = mask[zub:zlb:-1, yc, xlb:xub]
            out_shape = [(zub - zlb) * data_res[2], (xub - xlb) * data_res[0]]
            if out_shape[0] > out_shape[1]:
                out_shape = [unit, int(unit * out_shape[1] / out_shape[0])]
            else:
                out_shape = [int(unit * out_shape[0] / out_shape[1]), unit]
            out_slice = make_sshot(img_slice, mask_slice, out_shape, win_center, win_width, cm, k)
            canvas[i * unit: i * unit + out_shape[0], j * unit: j * unit + out_shape[1], :] = out_slice * 255.0
            j += 1

        if b_sag:
            img_slice = image[zub:zlb:-1, ylb:yub, xc]
            mask_slice = mask[zub:zlb:-1, ylb:yub, xc]
            out_shape = [(zub - zlb) * data_res[2], (yub - ylb) * data_res[1]]
            if out_shape[0] > out_shape[1]:
                out_shape = [unit, int(unit * out_shape[1] / out_shape[0])]
            else:
                out_shape = [int(unit * out_shape[0] / out_shape[1]), unit]
            out_slice = make_sshot(img_slice, mask_slice, out_shape, win_center, win_width, cm, k)
            canvas[i * unit: i * unit + out_shape[0], j * unit: j * unit + out_shape[1], :] = out_slice * 255.0
            j += 1

    return canvas
