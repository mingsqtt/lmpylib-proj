import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def plt(img, brg_to_rgb=True, equalize=False):
    plt.axis('off')
    if np.size(img.shape) == 3:
        if brg_to_rgb:
            if equalize:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif equalize:
            plt.imshow(img, vmin=0, vmax=255)
        else:
            plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()


def hist(img, bins=50, bychannel=True):
    x_scales = [0, 50, 100, 150, 200, 255]
    if type(img) == list or np.size(img.shape) == 1:
        plt.figure()
        nimg = len(img)
        for ind, img0 in enumerate(img):
            plt.subplot(math.ceil(nimg / 2), 2, ind+1)
            plt.xticks(x_scales)
            plt.yticks([])
            plt.hist(img0.flatten(), bins=bins, color="gray", range=(0, 255))
    elif np.size(img.shape) == 3 and bychannel:
        plt.xticks(x_scales)
        plt.yticks([])
        plt.hist(img[:, :, 2].flatten(), bins=bins, color="red", alpha=0.4, range=(0, 255))
        plt.hist(img[:, :, 1].flatten(), bins=bins, color="green", alpha=0.4, range=(0, 255))
        plt.hist(img[:, :, 0].flatten(), bins=bins, color="#72aee6", alpha=0.4, range=(0, 255))
    else:
        plt.xticks(x_scales)
        plt.yticks([])
        plt.hist(img.flatten(), bins=bins, color="gray", range=(0, 255))


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


def linear_comb_conv(img_patch, kernel):
    return(np.sum(img_patch * kernel))


def median_smooth_conv(img_patch, kernel):
    return(np.median(img_patch))


def _conv_channel(img_ch, kernel, conv_func, stripe, padding, padding_with, dtype):
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

                img_copy[h, w] = conv_func(img_patch, kernel)
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

                img_copy[h_target, w_target] = conv_func(img_patch, kernel)
    else:
        print("Invalid padding")
    return (img_copy)


def conv2D(img, kernel, conv_func=linear_comb_conv, stripe=1, padding="SAME", padding_with=0.0, normalize=False, dtype=np.int16):
    img_out = None
    if len(img.shape) == 2:
        img_out = _conv_channel(img, kernel, conv_func, stripe, padding, padding_with, dtype)
    elif len(img.shape) == 3:
        img_out = cv2.merge((
            _conv_channel(img[:, :, 0], kernel, conv_func, stripe, padding, padding_with, dtype),
            _conv_channel(img[:, :, 1], kernel, conv_func, stripe, padding, padding_with, dtype),
            _conv_channel(img[:, :, 2], kernel, conv_func, stripe, padding, padding_with, dtype)))
    else:
        print("Invalid shape for img")
        return(None)

    if normalize:
        return((255.0 * (img_out - np.min(img_out)) / (np.max(img_out) - np.min(img_out) + 0.01)).astype(np.uint8))
    else:
        return(img_out)


def median_blue(img, kernel_size=3):
    return(conv2D(img, np.zeros((kernel_size, kernel_size)), conv_func=median_smooth_conv, dtype=np.uint8))


def mean_blue(img, kernel_size=3):
    return(conv2D(img, np.ones((kernel_size, kernel_size)) / kernel_size**2, conv_func=linear_comb_conv, dtype=np.uint8))


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
