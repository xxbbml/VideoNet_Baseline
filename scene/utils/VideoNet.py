import torch.utils.data as data

import os
import sys
import random
import numpy as np
from PIL import Image
import json

def make_dataset(dir, class_to_idx):
    
    
    images = []
    image_list = open(dir, 'r')
    #dir = os.path.expanduser(dir)
 
    for image in image_list:
        path = image.rstrip().split(' ')[0]
        if not os.path.isfile(path):
            print('Can not read image %s' % path)
        else:
            target = image.rstrip().split(' ')[1]
            item = (path, class_to_idx[target])
            images.append(item)

    return images


class VideoNet(data.Dataset):

    def __init__(self,
                 root,
                 loader = None,
                 transform=None,
                 target_transform=None):
        
  
        self.loader = default_loader
        class_to_idx = json.load(open('class_to_idx.json','r'))

        samples = make_dataset(root, class_to_idx)
        
        self.root = root
        #self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
    
   
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    #with open(path, 'rb') as f:
    img = Image.open(path)
    return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
