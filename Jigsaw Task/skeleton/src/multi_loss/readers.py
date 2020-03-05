# from _future_ import print_function
# from _future_ import division
# from _future_ import unicode_literals
# from _future_ import absolute_import
import simplejpeg
import os
import os.path as pt
import zipfile
import re
from collections import defaultdict
import gzip

import numpy as np
import cv2
import msgpack

from datadings.reader import Reader
from datadings.sets._types import UnsupervisedImageData as YFCC100mData
from datadings.sets.YFCC100m_counts import FILE_COUNTS
from datadings.sets.YFCC100m_counts import FILES_TOTAL

ROOT = pt.abspath(pt.dirname(__file__))


def noop(data):
    return data


def validate_image(data):
    if len(data) < 2600 or len(data) == 9218:
        return None
    try:
        # decode at reduced scale for speedup
        im = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_REDUCED_GRAYSCALE_8
        )
        # image did not decode properly
        if im is None:
            return None
        # too little data, check for meaningful content
        if len(data) < 20000 and np.percentile(im.var(0), 95) < 50:
            # print()
            # print(len(data), np.percentile(im.var(0), 95))
            return None
        return data
    except cv2.error as e:
        print(e)
        if '!buf.empty() && buf.isContinuous() in function imdecode_' in str(e):
            return None
        else:
            raise e


def _find_zip_key(zips, key):
    z, f = key.split(os.sep)
    try:
        return zips.index(z), f
    except ValueError:
        raise IndexError('ZIP file {!r} not found'.format(z))


def _find_zip_index(rejects, index):
    total = FILES_TOTAL
    for z, _ in FILE_COUNTS:
        total -= len(rejects[z])
    rem = index
    if index < 0:
        rem += total
    if rem < 0 or rem >= total:
        raise IndexError('index {} out of range for {} items'.format(
            index, total - 1
        ))
    for i, (z, count) in enumerate(FILE_COUNTS):
        count -= len(rejects[z])
        if count > rem:
            return i, rem
        rem -= count


def _filter_zipinfo(infos):
    p = re.compile(r'/[0-9a-f]+$')
    return [info for info in infos if p.search(info.filename)]


def _find_member_image(members, rejected, start_image):
    if not start_image:
        return members
    for i, m in enumerate(members):
        if m.filename.split(os.sep)[1] == start_image:
            if i in rejected:
                raise IndexError(
                    '{!r} is on the rejected list'.format(m.filename)
                )
            return i
    raise IndexError('{!r} not found'.format(start_image))


def _find_member_index(rejected, start_index):
    for r in rejected:
        if start_index > r:
            start_index += 1
        else:
            break
    return start_index


def _find_start(
        path,
        rejected,
        start_key='',
        start_index=0
):
    if start_index and start_key:
        raise ValueError('cannot set both start_key and start_index')

    # collect all zips names in this object
    zips = [f for f, _ in FILE_COUNTS]
    # find out which zipfile to start from
    if start_index:
        zip_index, start_index = _find_zip_index(rejected, start_index)
        start_image = ''
    elif start_key:
        zip_index, start_image = _find_zip_key(zips, start_key)
    else:
        return zips, 0

    # now we got the correct zip index to start from 
    z = zips[zip_index]
    r = rejected[z]
    with zipfile.ZipFile(pt.join(path, z) + '.zip') as imagezip:
        # z must be bytes so the set of rejected images is found in py3
        # filter out non-image members
        members = _filter_zipinfo(imagezip.infolist())
    # start index also has to be set
    if start_index:
        start_index = _find_member_index(r, start_index)
        #print('the returned start_index is', start_index)
    elif start_image:
        start_index = _find_member_image(members, r, start_image)#
    #this part has been changed, now return the zip from the start_index 
    return zips[zip_index:], start_index


def yield_from_zips(
        path,
        zips,
        rejected,
        start_index,
        validator=noop,
):
    for z in zips:
        with zipfile.ZipFile(pt.join(path, z) + '.zip') as imagezip:
            r = rejected[z]
            # filter out non-image members
            members = _filter_zipinfo(imagezip.infolist())
            for i, m in enumerate(members[start_index:], start_index):
                if i in r:
                    continue
                f = m.filename
                yield validator(imagezip.read(f)), f, z, i
        start_index = 0


def _parse_rejected(f, rejected):
    new_rejected = msgpack.load(f, encoding='utf-8')
    for z, r in new_rejected.items():
        rejected[z].update(r)
    return rejected


class DevNull(object):
    def read(self, *_):
        pass

    def write(self, *_):
        pass

    def close(self):
        pass


class YFCC100mReader(Reader):
    def __init__(
            self,
            image_packs_dir = '/ds2/YFCC100m/image_packs',
            validator=validate_image,
            reject_file_paths=(
                    pt.join(ROOT, '../../data/YFCC100m_rejected_images.msgpack.gz'),
            ),
            error_file=None,
            error_file_mode='a',
    ):
        self._path = image_packs_dir
        if not callable(validator):
            raise ValueError('validator must be callable, not %r'
                             % validator)
        self._validator = validator
        self._next_sample = None
        self.count = 0
        self._rejected = defaultdict(lambda: set())
        try:
            for path in reject_file_paths:
                with gzip.open(path, 'rb') as f:
                    self._rejected = _parse_rejected(f, self._rejected)
        except IOError:
            pass
        if error_file is None:
            self._error_file = DevNull()
        else:
            self._error_file = open(error_file, error_file_mode)
        zips, start_index = _find_start(image_packs_dir, self._rejected)
        # this is the generator object which is used to get images
        self._gen = yield_from_zips(
            image_packs_dir, zips, self._rejected, start_index,
            self._validator,
        )
        
        self._packer = msgpack.Packer(
            use_bin_type=True, encoding='utf8'
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        if hasattr(self, '_error_file'):
            self._error_file.close()

    def __len__(self):
        return FILES_TOTAL - sum(len(r) for r in self._rejected.values())

    def _get_next_sample(self):
        while self._next_sample is None:
            sample, key, z, i = next(self._gen)
            if i not in self._rejected[z]:
                if sample is None:
                    self._rejected[z].add(i)
                    self._error_file.write('%s %d\n' % (z, i))
                else:
                    self._next_sample = sample, key
        return self._next_sample

    def next(self):
        self.count +=1
        sample = YFCC100mData(*self._get_next_sample())
        self._next_sample = None
        return sample

    __next__ = next

    def rawnext(self):
        return self._packer.pack(self.next())

    def seek_index(self, index):
        
        zips, start_index = _find_start(
            self._path, self._rejected, start_index=index
        )
        self._gen = yield_from_zips(
            self._path, zips, self._rejected, start_index, self._validator,
        )
        
    seek = seek_index

    def seek_key(self, key):
        zips, start_index = _find_start(
            self._path, self._rejected, start_key=key
        )
        self._gen = yield_from_zips(
            self._path, zips, self._rejected, start_index, self._validator,
        )

        print(len(self._gen))
    def get_key(self, index=None):
        return self._get_next_sample()[1]
