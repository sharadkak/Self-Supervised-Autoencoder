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
from crumpets.rng import MixtureRNG, INTERP_LINEAR

SCALE = 106/96
rnd_parameters = MixtureRNG(
    prob=0.5,
    scale_range=(1* SCALE, 1.25 * SCALE),
    shift_range=(-1, 1),
    # noise_range=(0.03, 0.1),
    noise_range=None,
    brightness_range=(-0.5, 0.5),
    color_range=(-0.5, 0.5),
    contrast_range=(-1, 1),
    # blur_range=(0.01, 0.75 / 224),
    blur_range=None,
    rotation_sigma=15,
    aspect_sigma=0.1,
    interpolations=(INTERP_LINEAR,),
    hmirror=0.5,
    vmirror=0,
    shear_range=(-0.1, 0.1),
)


ROOT = pt.abspath(pt.dirname(__file__)) + '/'


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


def get_tile_images(image, width=192, height=192):
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


def save_image(image, dir=None , name= None):
    '''save PIL images from numpy arrays'''

    image = PIL.Image.fromarray(image)
    image_save = pt.join( 'image_{}.png'.format(name))
    image.save(image_save)


def crop_tiles(tiles):

    new_tiles = np.zeros( shape = ( 9, 96, 96, 3),dtype = 'uint8' )
    for i in range(9):
        x, y = np.random.randint(0, 96), np.random.randint(0, 96)
        new_tiles[i] = tiles[i][x:x + 96, y:y + 96]
    return new_tiles


class ImageWorker(BufferWorker):
    """
    Worker for processing images of any kind.

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
        randomize_image_shape = (576, 576)

        image = cvt(randomize_image(
            im, randomize_image_shape,
            background=flat(self.fill_values[key]),
            **params
        ))

        # creating tiles from the image with shape 106x106
        tiles = get_tile_images(image)
        
        _, p, h, w, c = tiles.shape
        tiles = tiles.reshape((p * p, h, w, c))

        # applying random augmentations on the tiles
        rng = rnd_parameters(None, None)
        tiles = np.array([randomize_image(tiles[i], (96,96), flat(self.fill_values[key]), True, **rng) for i in range(p * p)])

        # random crop tiles of 96x96x3
        #tiles = crop_tiles(tiles)

        tiles = np.array([hwc2chw(tiles[i]) for i in range(p * p)])

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
