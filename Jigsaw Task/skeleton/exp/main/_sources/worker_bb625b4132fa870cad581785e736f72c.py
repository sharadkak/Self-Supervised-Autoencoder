from __future__ import print_function, division

import cv2
import numpy as np
import os
import os.path as pt
import PIL
from crumpets.broker import BufferWorker
from crumpets.augmentation import decode_image
from crumpets.presets import NO_AUGMENTATION
from crumpets.augmentation import randomize_image
from crumpets.augmentation import rotate_and_resize
from torchvision.utils import save_image
import itertools
import json

ROOT = pt.abspath(pt.dirname(__file__)) + '/'

# permutation set is created as per the algorithm given in jigsaw paper.
permutation_set = None
with open('file.txt', 'r') as f:
    permutation_set = json.load(f)


__all__ = [
    'ImageWorker',
    'ClassificationWorker',
    'FCNWorker'
]


def noop(im):
    return im


# noinspection PyUnresolvedReferences
def make_cvt(code):
    return lambda im: cv2.cvtColor(im, code)


# noinspection PyUnresolvedReferences
COLOR_CONVERSIONS = {
    None: noop,
    False: noop,
    '': noop,
    'rgb': noop,
    'RGB': noop,
    'hsv': make_cvt(cv2.COLOR_RGB2HSV_FULL),
    'HSV': make_cvt(cv2.COLOR_RGB2HSV_FULL),
    'hls': make_cvt(cv2.COLOR_RGB2HLS_FULL),
    'HLS': make_cvt(cv2.COLOR_RGB2HLS_FULL),
    'lab': make_cvt(cv2.COLOR_RGB2LAB),
    'LAB': make_cvt(cv2.COLOR_RGB2LAB),
    'ycrcb': make_cvt(cv2.COLOR_RGB2YCrCb),
    'YCrCb': make_cvt(cv2.COLOR_RGB2YCrCb),
    'YCRCB': make_cvt(cv2.COLOR_RGB2YCrCb),
    'gray': make_cvt(cv2.COLOR_RGB2GRAY),
    'GRAY': make_cvt(cv2.COLOR_RGB2GRAY),
}


def hwc2chw(im):
    return im.transpose((2, 0, 1))


def chw2hwc(im):
    return im.transpose((1, 2, 0))


def flat(array):
    return tuple(array.flatten().tolist())


def get_tile_images(image, width=106, height=106):
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )

# generate


def random_permutation():
    index = np.random.randint(0, 500)
    value = permutation_set[index]
    return value, index


def save_image(image, dir='../res/saved_test_input', name=1):
    '''save PIL images from numpy arrays'''
    dir = pt.join(ROOT, dir)
    image = PIL.Image.fromarray(image)
    image_save = pt.join(dir, 'image_%02d.png' % name)
    image.save(image_save)


def crop_tiles(tiles):

    new_tiles = np.array(np.random.randint(
        0, 255, size=(9, 96, 96, 3), dtype='uint8'))
    for i in range(9):
        x, y = np.random.randint(0, 10), np.random.randint(0, 10)
        new_tiles[i] = tiles[i][x:x + 96, y:y + 96]
    return new_tiles


def permutate_tiles(tiles, permute):
    '''Jumble images tiles as per provided permutation'''

    blocks = {i: tiles[i] for i in range(3 * 3)}
    new_tiles = np.array([blocks[value]
                          for i, value in enumerate(permute)])
    return new_tiles


class ImageWorker(BufferWorker):
    """
    Worker for processing images of any kind.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param image_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    :param gpu_augmentation:
        disables augmentations for which
        gpu versions are available (:class:`~crumpets.torch.randomizer`)
    """

    def __init__(self, image,
                 image_params=None,
                 image_rng=None,
                 **kwargs):
        BufferWorker.__init__(self, **kwargs)
        self.add_buffer('image', image)
        self.add_params('image', image_params, {})
        self.image_rng = image_rng or NO_AUGMENTATION

    # this fn is being called twice, first for image then for target image processing
    # this fn will preprocess the image and assignd it to the key in buffers
    def prepare_image(self, im, buffers, params, key):
        # im image shape is 500x300x3
        params = dict(params)
        params.update(self.params[key])

        # change the color of the image
        cvt = COLOR_CONVERSIONS[params.pop('colorspace', None)]
        # change the image format and apply random augmentation to it given in the params
        # Here we are not using any parameters as params has mostly nothing
        # first get 318x318 image and then divide them in 106x106 tiles
        randomize_image_shape = (318, 318)

        image = cvt(randomize_image(
            im, randomize_image_shape,
            background=flat(self.fill_values[key]),
            **params
        ))

        # creating tiles from the image with shape 106x106
        tiles = get_tile_images(image)

        _, p, h, w, c = tiles.shape
        tiles = tiles.reshape((p * p, h, w, c))

        # random crop tiles of 96x96x3
        tiles = crop_tiles(tiles)

        # save some tiles for inspection
        # for i in range(tiles.shape[0]):
        #     save_image(tiles[i], name = i+1)

        tiles = np.array([hwc2chw(tiles[i]) for i in range(p * p)])
        # here permute the tiles randomly and check if its working by saving them as png images
        permute, index = random_permutation()

        # probably write some list comprehension to permute first dim of tiles tensor
        tiles = permutate_tiles(tiles, permute)
        # generate ground truth for classifier
        buffers['label'][...] = index

        # Now tiles shape will be 9x3,96,96 and this will be used to create a batch
        p, c, h, w = tiles.shape
        tiles = (tiles.reshape((9 * c, h, w)))
        # save the processed to the buffer with key
        buffers[key][...] = tiles
        buffers['target_image'][...] = tiles

        # this is the processed image stored in buffer as numpy object of shape 3x224x224
        return params

    def prepare(self, sample, batch, buffers):
        # sample contains the image in bytes
        im = decode_image(sample['image'],
                          self.params['image'].get('color', True))
        # now it's decoded and converted to numpy array

        # this applies augmentation but not so useful for us
        params = self.image_rng(im, buffers['image'])
        # this is the only params we have
        params['gpu_augmentation'] = self.gpu_augmentation
        image_params = self.prepare_image(im, buffers, params, 'image')
        batch['augmentation'].append(image_params)

        return im, params


class ClassificationWorker(ImageWorker):
    """
    Worker for processing (Image, Label)-pairs for classification.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param label:
        tuple of label information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param image_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    """

    def __init__(self, image, label,
                 image_params=None,
                 image_rng=None,
                 **kwargs):
        ImageWorker.__init__(self, image,
                             image_params,
                             image_rng,
                             **kwargs)
        self.add_buffer('label', label)

    def prepare(self, sample, batch, buffers):
        im, params = ImageWorker.prepare(self, sample, batch, buffers)
        buffers['label'][...] = sample['label']
        return im, params


class FCNWorker(ImageWorker):
    """
    Worker for fully convolutional networks (FCN).
    Produces `image`-`target_image`-pairs.

    :param image:
        tuple of image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param target_image:
        tuple of target image information (shape, dtype, fill_value);
        fill_value is optional, defaults to 0
    :param image_params:
        dict of fixed image parameters;
        overwrites random augmentation values
    :param target_image_params:
        dict of fixed target image parameters;
        overwrites random augmentation values
    :param image_rng:
        RNG object used for image augmentation,
        see :class:`~crumpets.rng.RNG` and
        :func:`~crumpets.randomization.randomize_args`
    """

    def __init__(self, image, target_image,
                 image_params=None, target_image_params=None,
                 image_rng=None,
                 **kwargs):
        ImageWorker.__init__(self, image,
                             image_params,
                             image_rng,
                             **kwargs)
        self.add_buffer('target_image', target_image)
        self.add_params('target_image', target_image_params, {})
        label = ((1,), np.int)
        self.add_buffer('label', label)

    def prepare(self, sample, batch, buffers):
        # buffer is being passed from the process function of BufferWorker
        # sample dict has image and key
        # batch is a dictionary containing list with informaton about augmentation

        # got an image from memory, in bytes
        # input image is being processed
        im, params = ImageWorker.prepare(
            self, sample, batch, buffers)  # shape 334 x 500 x 3

        # target image is being processed.
        #self.prepare_image(im, buffers, params, 'target_image')
