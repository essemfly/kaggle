import cv2
import matplotlib.pyplot as plt, numpy as np
from progressbar import ProgressBar


def clahe_light(image, clipLimit=2.0, tileGridSize=(8, 8), blur=False):
    """ Returns images to which Contrast Limited Adaptive Histogram Equalization is applied
    Args:
        image: 3d-array of type uint8, BGR color image
        clipLimit: float, parameter for CLAHE which determines local contrast limitation
        tileGridSize: (int, int), size of local region in which histogram equalization is performed
        blur: boolean, determines whether 3x3 Gaussian blur to be applied to output images
    Returns:
        An image after CLAHE, same type as image
    """
    # Gaussian blur
    if blur:
        _image = cv2.GaussianBlur(image, (3, 3), 0)
    else:
        _image = image

    img_lab = cv2.cvtColor(_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    l = clahe.apply(l)
    img_after = cv2.merge((l, a, b))
    img_after = cv2.cvtColor(img_after, cv2.COLOR_LAB2BGR)
    return img_after


def clahe_channel(image, clipLimit=2.0, tileGridSize=(8, 8), blur=False):
    """ Returns images to which Contrast Limited Adaptive Histogram Equalization is applied
    Args:
        image: 3d-array of type uint8, BGR color image
        clipLimit: float, parameter for CLAHE which determines local contrast limitation
        tileGridSize: (int, int), size of local region in which histogram equalization is performed
        blur: boolean, determines whether 3x3 Gaussian blur to be applied to output images
    Returns:
        An image after CLAHE, same type as image
    """
    # Gaussian blur
    if blur:
        _image = cv2.GaussianBlur(image, (3, 3), 0)
    else:
        _image = image

    channels = cv2.split(_image)
    channels_clahe = []
    for c in channels:
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        channels_clahe.append(clahe.apply(c))
    img_after = cv2.merge(channels_clahe)

    return img_after


def subtract_local_mean(image, amplify=4, kernel_size=7):
    """ Returns images to which pixelwise local mean subtract is applied
    Args:
        image: 3d-array of type uint8, BGR color image
        amplify: int, how much differences from mean pixel to be amplified
        kernel_size: int, kernel size for Gaussian flitering
    Returns:
        An image before processed
        An image after processed
    """
    # Contrast enhancement
    img_after = cv2.addWeighted(image, amplify,
                                cv2.GaussianBlur(image, (kernel_size, kernel_size), 0), -amplify, 128)
    return img_after


def subtract_local_light(image, amplify=4, kernel_size=7):
    """ Returns images to which brightness adjustment is applied
    Args:
        image: 3d-array of type uint8, BGR color image
        amplify: int, how much differences from mean pixel to be amplified
        kernel_size: int, kernel size for Gaussian flitering
    Returns:
        image_after: same type with image. BGR color image
    """
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    l = cv2.addWeighted(l, amplify,
                        cv2.GaussianBlur(l, (kernel_size, kernel_size), 0), -amplify, 128)
    img_after = cv2.merge((l, a, b))
    img_after = cv2.cvtColor(img_after, cv2.COLOR_LAB2BGR)
    return img_after


def preview(path, process_fn, show_now=False, **kwargs_list):
    """ Preview results after CLAHE
    Args:
        path: String, path to an image file
        process_fn: function, local mean subtract function
        show_now: Boolean, whether to show this plot right now or wait when plt.show() called later
        kwargs_list: Keyword arguments each is a list of argument values. \
        e.g. keyword_arg1 = [value1, value2], keyword_arg2 = [value1, value2]
    Returns:
        None
    """
    image = cv2.imread(path)
    subplot_imgs = [image]

    nb_args = [len(kwargs_list[k]) for k in kwargs_list.keys()]
    if len(set(nb_args)) != 1:
        raise ValueError('Each keyword arguments must be lists of same length')

    for i in range(nb_args[0]):
        kwargs = dict([(k, kwargs_list[k][i]) for k in kwargs_list.keys()])

        img_after = process_fn(image, **kwargs)  # default parameters
        subplot_imgs.append(img_after)

    plt.figure()
    for i in range(len(subplot_imgs)):
        plt.subplot(1, len(subplot_imgs), i + 1)
        plt.imshow(cv2.cvtColor(subplot_imgs[i], cv2.COLOR_BGR2RGB))

    if show_now:
        plt.show()
