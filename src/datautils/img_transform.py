import numpy as np
import cv2

from skimage.transform import resize


def affine_transformation(img, m=1.0, s=.2, border_value=None):
    h, w = img.shape[0], img.shape[1]
    src_point = np.float32([[w / 2.0, h / 3.0],
                            [2 * w / 3.0, 2 * h / 3.0],
                            [w / 3.0, 2 * h / 3.0]])
    random_shift = m + np.random.uniform(-1.0, 1.0, size=(3,2)) * s
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if border_value is None:
        border_value = np.median(img)
    warped_img = cv2.warpAffine(img, transform, dsize=(w, h), borderValue=float(border_value))
    return warped_img


def apply_preprocessing_multiscale(img, fheight, fwidth):

    temp_img = img
    # Handwriting in white, background in black
    temp_img = 1 - temp_img.astype(np.float32) / 255.0

    h, w = temp_img.shape

    h_threshold_small = fheight * 0.75
    w_threshold_small = fwidth * 0.75

    # Soft resize for small image
    # Upscale small image
    if h < h_threshold_small or w < w_threshold_small:
        scale_y = h_threshold_small / float(h)
        scale_x = w_threshold_small / float(w)

        scale = min(scale_x, scale_y)

        width_new = int(scale * w)
        height_new = int(scale * h)

        temp_img = resize(image=temp_img, output_shape=(height_new, width_new)).astype(np.float32)

    # Down scale large image
    if h > fheight or w > fwidth:
        scale_y = float(fheight) / float(h)
        scale_x = float(fwidth) / float(w)

        scale = min(scale_x, scale_y)

        width_new = int(scale * w)
        height_new = int(scale * h)

        temp_img = resize(image=temp_img, output_shape=(height_new, width_new)).astype(np.float32)

    temp_img = centered_img(temp_img, (fheight, fwidth), border_value=0.0)

    temp_img = np.pad(temp_img, ((0, 0), (int(fheight / 4), fheight * 2)), 'constant', constant_values=0)

    return temp_img


# From: https://github.com/georgeretsi/HTR-best-practices/
def centered_img(word_img, tsize, centering=(.5, .5), border_value=None):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)

    word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=border_value)
    return word_img


def pad_images_otherdim(data, padding_value):
    """
    data: list of numpy array
    """
    x_lengths = [x.shape[2] for x in data]
    y_lengths = [x.shape[1] for x in data]
    longest_x = max(x_lengths)
    longest_y = max(y_lengths)

    nb_channel = data[0].shape[0]
    padded_data = np.ones((len(data), nb_channel, longest_y, longest_x )) * padding_value
    for i, xy_len in enumerate(zip(x_lengths, y_lengths)):
        x_len, y_len = xy_len
        padded_data[i, :, :y_len, :x_len] = data[i][:, :y_len, :x_len]
    return padded_data

