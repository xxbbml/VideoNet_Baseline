import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2
import scipy.io as scio

class VideoNet(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality="rbg",
                 name_pattern=None,
                 is_color=True,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, source, class_to_idx,phase)

        print(len(clips))
        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "%d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d.png"

        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.clips[index]

        clip_input = ReadFrames(path,
                                self.new_height,
                                self.new_width,
                                self.name_pattern
                                )


        if self.transform is not None:
            clip_input = self.transform(clip_input)

        return clip_input, target 


    def __len__(self):
        return len(self.clips)