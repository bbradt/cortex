'''Module for handling neuroimaging data saved in .mat files
 We build an "ImageFolder" object and we can iterate/index through
 it.  The class is initialized with a folder location, a loader (the only one
    we have now is for nii files), and (optionally) a list of regex patterns.

 The user can also provide a 3D binary mask (same size as data) to vectorize
 the space/voxel dimension. Can handle 3D and 3D+time (4D) datasets So, it can
 be built one of two ways:
 1: a path to one directory with many images, and the classes are based on
 regex patterns.
   example 1a: "/home/user/some_data_path" has files *_H_*.nii and *_S_*.nii
               files
  patterned_images = ImageFolder("/home/user/some_data_path",
                     patterns=['*_H_*','*_S_*'] , loader=nii_loader)
    example 1b: "/home/user/some_data_path" has files *_H_*.nii and *_S_*.nii
    files, and user specifies a mask to vectorize space
  patterned_images_mask = ImageFolder("/home/user/some_data_path",
    patterns=['*_H_*','*_S_*'] , loader=nii_loader,
              mask="/home/user/maskImage.nii")

 2: a path to a top level directory with sub directories denoting the classes.
    example 2a: "/home/user/some_data_path" has subfolders 0 and 1 with nifti
    files corresponding to class 0 and class 1 respectively
  foldered_images = ImageFolder("/home/user/some_data_path",loader=nii_loader)
    example 2b: Same as above but with a mask
  foldered_images = ImageFolder("/home/user/some_data_path",loader=nii_loader,
                                mask="/home/user/maskImage.nii")


 The final output (when we call __getitem__) is a tuple of: (image,label)
'''

import torch.utils.data as data

import os
import os.path
import numpy as np
from glob import glob
import scipy.io as sio

IMG_EXTENSIONS = ['.mat']
DEFAULT_data_keys = ['x1']
DEFAULT_label_key = 'y'

def make_dataset(dir, patterns=None):
    """

    Args:
        dir:
        patterns:

    Returns:

    """
    images = []

    dir = os.path.expanduser(dir)

    file_list = []

    all_items = [os.path.join(dir, i) for i in os.listdir(dir)]
    directories = [os.path.join(dir, d) for d in all_items if os.path.isdir(d)]
    if patterns is not None:
        for i, pattern in enumerate(patterns):
            files = [(f, i) for f in glob(os.path.join(dir, pattern))]
            file_list.append(files)
    else:
        file_list = [[(os.path.join(p, f), i)
                      for f in os.listdir(p)
                      if os.path.isfile(os.path.join(p, f))]
                     for i, p in enumerate(directories)]

    for i, target in enumerate(file_list):
        for item in target:
            images.append(item)

    return images


class ImageFolder(data.Dataset):
    '''
    Args:
        root (string): Root directory path.
        patterns (list): list of regex patterns
        loader (callable, optional): A function to load an image given its
        path.

     Attributes:
        imgs (list): List of (image path, class_index) tuples
    '''

    def __init__(self, root, loader=sio.loadmat, patterns=None,
                 data_keys=DEFAULT_data_keys,
                 label_key=DEFAULT_label_key, transform=None):
        imgs = make_dataset(root, patterns)

        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " +
                    root +
                    "\n"
                    "Supported image extensions are: " +
                    ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.images = imgs
        self.transform = transform
        self.data_keys = data_keys
        self.label_key = label_key

        self.loader = loader

    '''
        Gives us a tuple from the array at (index) of: (image, label)
    '''

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target
            class.
        """
        label = self.imgs[index][self.label_key]
        img = np.array([self.imgs[index][key] for key in self.data_keys])
        if self.transform is not None:
            img = self.transform(img)

        return np.array(img), label

    def __len__(self):
        return len(self.images)
