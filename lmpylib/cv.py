import numpy as np
import matplotlib.pyplot as pyplot
import cv2
import math
import itertools


def plt(img, brg_to_rgb=True, equalize=False):
    pyplot.axis('off')
    if np.size(img.shape) == 3:
        if brg_to_rgb:
            if equalize:
                pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
            else:
                pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif equalize:
            pyplot.imshow(img, vmin=0, vmax=255)
        else:
            pyplot.imshow(img)
    else:
        pyplot.imshow(img, cmap="gray", vmin=0, vmax=255)
    pyplot.show()


def hist(img, bins=50, bychannel=True):
    x_scales = [0, 50, 100, 150, 200, 255]
    if type(img) == list or np.size(img.shape) == 1:
        pyplot.figure()
        nimg = len(img)
        for ind, img0 in enumerate(img):
            pyplot.subplot(math.ceil(nimg / 2), 2, ind+1)
            pyplot.xticks(x_scales)
            pyplot.yticks([])
            pyplot.hist(img0.flatten(), bins=bins, color="gray", range=(0, 255))
    elif np.size(img.shape) == 3 and bychannel:
        pyplot.xticks(x_scales)
        pyplot.yticks([])
        pyplot.hist(img[:, :, 2].flatten(), bins=bins, color="red", alpha=0.4, range=(0, 255))
        pyplot.hist(img[:, :, 1].flatten(), bins=bins, color="green", alpha=0.4, range=(0, 255))
        pyplot.hist(img[:, :, 0].flatten(), bins=bins, color="#72aee6", alpha=0.4, range=(0, 255))
    else:
        pyplot.xticks(x_scales)
        pyplot.yticks([])
        pyplot.hist(img.flatten(), bins=bins, color="gray", range=(0, 255))


def shift(img, offset_shape, output_shape=None):
    if output_shape is None:
        output_shape = img.shape
    matrix = np.float32([
        [1, 0, offset_shape[1]],
        [0, 1, offset_shape[0]]])
    return(cv2.warpAffine(img, matrix, (output_shape[1], output_shape[0])))


def rotate(img, angle_deg, center_coor=None, output_shape=None, fit_frame=False):
    height = img.shape[0]
    width = img.shape[1]
    angle_rad = angle_deg * math.pi / 180
    if center_coor is None:
        center_coor = (width / 2, height / 2)
    if output_shape is None:
        if fit_frame:
            diag = (width**2 + height**2)**0.5
            sin_a = math.sin(angle_rad % (math.pi/2))
            cos_a = math.cos(angle_rad % (math.pi/2))
            sin_b = height / diag
            cos_b = width / diag
            output_height = round(diag * (sin_a * cos_b + sin_b * cos_a))
            output_width = round(diag * (cos_a * cos_b + sin_a * sin_b))
            output_shape = (output_height, output_width)
        else:
            output_shape = img.shape
    matrix = cv2.getRotationMatrix2D(center_coor, angle_deg, 1)
    return(cv2.warpAffine(img, matrix, (output_shape[1], output_shape[0])))


def rotate_rad(img, angle_rad, center_coor=None, output_shape=None):
    if center_coor is None:
        center_coor = (img.shape[1] / 2, img.shape[0] / 2)
    if output_shape is None:
        output_shape = img.shape
    matrix = cv2.getRotationMatrix2D(center_coor, angle_rad * 360 / (math.pi * 2), 1)
    return(cv2.warpAffine(img, matrix, (output_shape[1], output_shape[0])))


def reduce_size(img, max_height_or_width=500):
    shape = img.shape
    if shape[0] >= shape[1]:
        if shape[0] > max_height_or_width:
            return(cv2.resize(img, (int(500*shape[1]/shape[0]), 500)))
        else:
            return(img)
    else:
        if shape[1] > max_height_or_width:
            return(cv2.resize(img, (500, int(500*shape[0]/shape[1]))))
        else:
            return(img)


def down_sample(data, upto, gaussion_prefilter_size=(5,5), gaussion_prefilter_sigma=None):
    old_shape = data.shape
    n_dim = len(old_shape)
    if (type(upto) == float) and (upto < 1.0):
        if n_dim == 3:
            percent = np.array([upto, upto, 1])
        elif n_dim == 4:
            percent = np.array([1, upto, upto, 1])
        else:
            percent = np.repeat(upto, n_dim)
    elif (type(upto) == int) and (upto > 1):
        max_side = np.max(np.array(old_shape))
        if max_side > upto:
            perc = upto / max_side
            if n_dim == 3:
                percent = np.array([perc, perc, 1])
            elif n_dim == 4:
                percent = np.array([1, perc, perc, 1])
            else:
                percent = np.repeat(perc, n_dim)
        else:
            return data
    elif (type(upto) == tuple) and (len(upto) == n_dim):
        percent = np.ones(n_dim)
        for dim in range(n_dim):
            if upto[dim] < 1.0:
                percent[dim] = upto[dim]
            elif old_shape[dim] > upto[dim]:
                percent[dim] = upto[dim] / old_shape[dim]
    else:
        raise Exception("upto must be a percentage between (0, 1), or length integer, or size tuple.")

    indices = np.indices(old_shape)
    down_sampling_filter = np.ones(old_shape, dtype=np.bool)

    below_half = percent <= 0.5
    above_half = (percent > 0.5) & (percent < 1.0)
    same_size = percent == 1.0
    denominator = np.empty(n_dim)
    denominator[below_half] = (np.round(1 / percent[below_half], 0)).astype(int)
    denominator[above_half] = (np.round(1 / (1 - percent[above_half]), 0)).astype(int)
    denominator[same_size] = 9999
    remainder = (denominator / 2).astype(int)
    new_shape = np.array(old_shape)
    new_shape[below_half] = (np.ceil((np.array(old_shape) - remainder) / denominator)).astype(int)[below_half]
    new_shape[above_half] = \
    (np.array(old_shape) - (np.ceil((np.array(old_shape) - remainder) / denominator)).astype(int))[above_half]

    for dim in range(n_dim):
        if below_half[dim]:
            down_sampling_filter = down_sampling_filter & (np.mod(indices[dim], denominator[dim]) == remainder[dim])
        elif above_half[dim]:
            down_sampling_filter = down_sampling_filter & (np.mod(indices[dim], denominator[dim]) != remainder[dim])

    if (gaussion_prefilter_sigma is None) or (gaussion_prefilter_size is None):
        return data[down_sampling_filter].reshape(new_shape)
    else:
        return cv2.GaussianBlur(data, ksize=gaussion_prefilter_size, sigmaX=gaussion_prefilter_sigma)[
            down_sampling_filter].reshape(new_shape)


def linear_comb_filter(img_patch, kernel):
    return(np.sum(img_patch * kernel))


def median_smooth_filter(img_patch, kernel):
    return(np.median(img_patch))


def _filter_channel(img_ch, kernel, filter_func, stripe, padding, padding_with, dtype):
    img_patch = np.zeros(kernel.shape, dtype)
    height = img_ch.shape[0]
    width = img_ch.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    kernel_height_offset = (int(kernel_height / 2), math.ceil(kernel_height / 2))
    kernel_width_offset = (int(kernel_width / 2), math.ceil(kernel_width / 2))
    if padding.lower() == "same":
        img_copy = np.zeros(img_ch.shape, dtype)
        for h in range(0, height, stripe):
            for w in range(0, width, stripe):

                for h0 in range(kernel_height):
                    for w0 in range(kernel_width):
                        img_h = h - kernel_height_offset[0] + h0
                        img_w = w - kernel_width_offset[0] + w0
                        if 0 <= img_h < height and 0 <= img_w < width:
                            img_patch[h0, w0] = img_ch[img_h, img_w]
                        else:
                            img_patch[h0, w0] = padding_with

                img_copy[h, w] = filter_func(img_patch, kernel)
    elif padding.lower() == "valid":
        img_copy = np.zeros((int((height - kernel_height) / stripe) + 1, int((width - kernel_width) / stripe) + 1), dtype)
        for h_target, h_src in enumerate(range(kernel_height_offset[0], height - kernel_height_offset[1] + 1, stripe)):
            for w_target, w_src in enumerate(range(kernel_width_offset[0], width - kernel_width_offset[1] + 1, stripe)):

                for h0 in range(kernel_height):
                    for w0 in range(kernel_width):
                        img_h = h_src - kernel_height_offset[0] + h0
                        img_w = w_src - kernel_width_offset[0] + w0
                        if 0 <= img_h < height and 0 <= img_w < width:
                            img_patch[h0, w0] = img_ch[img_h, img_w]
                        else:
                            img_patch[h0, w0] = padding_with

                img_copy[h_target, w_target] = filter_func(img_patch, kernel)
    else:
        print("Invalid padding")
    return (img_copy)


# img_ch = np.linspace(0, 29, 30, dtype=np.uint8).reshape(6, 5)
# hor = _filter_channel_1D_linear_combine(img_ch, cv2.getGaussianKernel(3, 1).flatten(), 0, 1, "same", 0, np.float)
# ver = _filter_channel_1D_linear_combine(img_ch, cv2.getGaussianKernel(3, 1).flatten(), 1, 1, "same", 0, np.float)
# combine = hor*ver
#
# filter2D(img_ch, gaussian_kernal(3, 1))

def _filter_channel_1D_linear_combine(img_ch, kernel_1D, axis, stripe, padding, padding_with, dtype):
    height = img_ch.shape[0]
    width = img_ch.shape[1]
    kernel_len = len(kernel_1D)
    kernel_len_offset = (int(kernel_len / 2), math.ceil(kernel_len / 2))
    if padding.lower() == "same":
        dot_with = np.zeros((width*height, kernel_len), dtype)
        img_copy = np.zeros(img_ch.shape, dtype)
        for h in range(0, height, stripe):
            for w in range(0, width, stripe):

                if axis == 0:
                    start = w - kernel_len_offset[0]
                    end = w + kernel_len_offset[1]
                    dw = img_ch[h, max(start, 0):min(end, width)]
                    if start < 0:
                        dw = np.concatenate([np.repeat(padding_with, -start), dw])
                    if end > width:
                        dw = np.concatenate([dw, np.repeat(padding_with, end - width)])
                    dot_with[h * width + w] = dw
                elif axis == 1:
                    start = h - kernel_len_offset[0]
                    end = h + kernel_len_offset[1]
                    dw = img_ch[max(start, 0):min(end, height), w].flatten()
                    if start < 0:
                        dw = np.concatenate([np.repeat(padding_with, -start), dw])
                    if end > height:
                        dw = np.concatenate([dw, np.repeat(padding_with, end - height)])
                    dot_with[w * height + h] = dw
                else:
                    raise Exception("axis must be 0 or 1")

        if axis == 0:
            return dot_with.dot(kernel_1D).reshape(height, width)
        elif axis == 1:
            return dot_with.dot(kernel_1D).reshape(width, height).transpose()

                # img_copy[h, w] = filter_func(img_patch, kernel_1D)
    # elif padding.lower() == "valid":
    #     img_copy = np.zeros((int((height - kernel_height) / stripe) + 1, int((width - kernel_width) / stripe) + 1), dtype)
    #     for h_target, h_src in enumerate(range(kernel_height_offset[0], height - kernel_height_offset[1] + 1, stripe)):
    #         for w_target, w_src in enumerate(range(kernel_width_offset[0], width - kernel_width_offset[1] + 1, stripe)):
    #
    #             for h0 in range(kernel_height):
    #                 for w0 in range(kernel_width):
    #                     img_h = h_src - kernel_height_offset[0] + h0
    #                     img_w = w_src - kernel_width_offset[0] + w0
    #                     if 0 <= img_h < height and 0 <= img_w < width:
    #                         img_patch[h0, w0] = img_ch[img_h, img_w]
    #                     else:
    #                         img_patch[h0, w0] = padding_with
    #
    #             img_copy[h_target, w_target] = filter_func(img_patch, kernel_1D)
    else:
        print("Invalid padding")


def filter2D(img, kernel, filter_func=linear_comb_filter, stripe=1, padding="SAME", padding_with=0.0, normalize=False, dtype=np.int16):
    if len(img.shape) == 2:
        img_out = _filter_channel(img, kernel, filter_func, stripe, padding, padding_with, dtype)
    elif len(img.shape) == 3:
        img_out = cv2.merge((
            _filter_channel(img[:, :, 0], kernel, filter_func, stripe, padding, padding_with, dtype),
            _filter_channel(img[:, :, 1], kernel, filter_func, stripe, padding, padding_with, dtype),
            _filter_channel(img[:, :, 2], kernel, filter_func, stripe, padding, padding_with, dtype)))
    else:
        print("Invalid shape for img")
        return(None)

    if normalize:
        return((255.0 * (img_out - np.min(img_out)) / (np.max(img_out) - np.min(img_out) + 0.01)).astype(np.uint8))
    else:
        return(img_out)


def median_blue(img, kernel_size=3):
    return(filter2D(img, np.zeros((kernel_size, kernel_size)), filter_func=median_smooth_filter, dtype=np.uint8))


def mean_blue(img, kernel_size=3):
    return(filter2D(img, np.ones((kernel_size, kernel_size)) / kernel_size**2, filter_func=linear_comb_filter, dtype=np.uint8))


def gaussian_kernal(size, sigma):
    kernal1d = cv2.getGaussianKernel(size, sigma).flatten()
    prod = np.array(list(itertools.product(kernal1d, kernal1d)))
    return np.apply_along_axis(lambda tup: tup[0] * tup[1], 1, prod).reshape(size, size)


def equalize_hist(img, bychannel=False):
    eq_copy = img.copy()
    if len(img.shape) == 3 and bychannel:
        for channel in range(3):
            img_ch = eq_copy[:, :, channel]
            lvls, freq = np.unique(img_ch, return_counts=True)
            if lvls[0] != 0:
                np.insert(lvls, 0, 0)
                np.insert(freq, 0, 0)
            if lvls[len(lvls) - 1] != 255:
                np.append(lvls, 255)
                np.append(freq, 0)
            total_pix = np.size(img_ch)
            Pr = freq / total_pix
            acc_pr = 0.0
            s = np.zeros(len(lvls))
            for ind, pr in enumerate(Pr):
                acc_pr = acc_pr + pr
                s[ind] = acc_pr
            new_lvls = np.round(s * 255.0).astype(np.uint8)
            for ind, old_lvl in enumerate(lvls):
                img_ch[img[:, :, channel] == old_lvl] = new_lvls[ind]
    else:
        lvls, freq = np.unique(eq_copy, return_counts=True)
        if lvls[0] != 0:
            np.insert(lvls, 0, 0)
            np.insert(freq, 0, 0)
        if lvls[len(lvls) - 1] != 255:
            np.append(lvls, 255)
            np.append(freq, 0)
        total_pix = np.size(eq_copy)
        Pr = freq / total_pix
        acc_pr = 0.0
        s = np.zeros(len(lvls))
        for ind, pr in enumerate(Pr):
            acc_pr = acc_pr + pr
            s[ind] = acc_pr
        new_lvls = np.round(s * 255.0).astype(np.uint8)
        for ind, old_lvl in enumerate(lvls):
            eq_copy[img == old_lvl] = new_lvls[ind]
    return(eq_copy)


def contour_centroid(contour):
    moments = cv2.moments(contour)
    return({
        "x": int(moments['m10']/moments['m00']),
        "y": int(moments['m01']/moments['m00'])})


def contour_content_points(img_gray, contour, use_cv_or_numpy="numpy"):
    mask = np.zeros(img_gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    if use_cv_or_numpy == "numpy":
        return(np.transpose(np.nonzero(mask)))
    else:
        return(cv2.findNonZero(mask))
