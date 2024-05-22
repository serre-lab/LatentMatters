import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as f
import math


class Binarize(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self, binary_threshold=0.5):
        self.binary_threshold = binary_threshold

    def __call__(self, sample):
        mask = torch.zeros_like(sample)
        mask[sample > self.binary_threshold] = 1

        return mask


class Scale_0_1(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self):
        self.te = 1

    def __call__(self, sample):
        #sample = sample - sample.min()
        #sample /= sample.max()
        return sample/255


class Binarize_batch(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self, binary_threshold=0.5):
        self.binary_threshold = binary_threshold

    def __call__(self, sample):
        mask = torch.zeros_like(sample)
        mask[sample > self.binary_threshold] = 1

        return mask


class Scale_0_1_batch(object):
    """ Binarize the input given a threshold on the pixel value.

    Args:
        binary_threshold:  all pixels with a value lower than the binary threshold are going to be 0, all others are 1.
    """

    def __init__(self):
        self.te = 1

    def __call__(self, sample):
        mini,_ = sample.view(sample.size(0), -1).min(dim=1)
        sample = sample - mini.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        maxi, _ = sample.view(sample.size(0), -1).max(dim=1)
        sample /= maxi.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return sample

