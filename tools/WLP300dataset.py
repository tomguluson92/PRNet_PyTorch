# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.18
    @readme: The implementation of PRNet Network DataLoader.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import cv2
import random
import numbers
import numpy as np
from PIL import Image
from skimage import io

data_transform = {'train': transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}


class PRNetDataset(Dataset):
    """Pedestrian Attribute Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dict = dict()
        self._max_idx()

    def get_img_path(self, img_id):
        img_id = self.dict.get(img_id)
        if img_id:
            original = os.path.join(self.root_dir, str(img_id), 'original.jpg')
            uv_map = os.path.join(self.root_dir, str(img_id), 'uv_posmap.jpg')

            return original, uv_map

    def _max_idx(self):
        _tmp_lst = map(lambda x: int(x), os.listdir(self.root_dir))
        _sorted_lst = sorted(_tmp_lst)
        for idx, item in enumerate(_sorted_lst):
            self.dict[idx] = item

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        original, uv_map = self.get_img_path(idx)

        origin = Image.open(original).convert("RGB")
        uv_map = Image.open(uv_map).convert("RGB")

        sample = {'uv_map': uv_map, 'origin': origin}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']
        uv_map, origin = np.array(uv_map), np.array(origin)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        uv_map = uv_map.transpose((2, 0, 1))
        origin = origin.transpose((2, 0, 1))

        uv_map = uv_map.astype("float32") / 255.
        origin = origin.astype("float32") / 255.
        return {'uv_map': torch.from_numpy(uv_map), 'origin': torch.from_numpy(origin)}


class ToNormalize(object):
    """Normalized process on origin Tensors."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']
        origin = F.normalize(origin, self.mean, self.std, self.inplace)
        return {'uv_map': uv_map, 'origin': origin}


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transform = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transform.append(transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transform.append(transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transform.append(transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transform.append(transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transform)
        transform = transforms.Compose(transform)

        return transform

    def __call__(self, sample):
        """
        Args:
            sample (PIL Image): {'uv_map': PIL Image, 'origin': PIL Image}.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        uv_map, origin = sample['uv_map'], sample['origin']
        origin = transform(origin)

        return {'uv_map': uv_map, 'origin': origin}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
