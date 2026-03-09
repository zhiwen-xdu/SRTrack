import os
import os.path
import torch
import numpy as np
import pandas
import csv
from glob import glob
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.event_utils import get_merge_frame


class FE108(BaseVideoDataset):
    """ FE108 dataset."""
    def __init__(self, root=None, split='train', image_loader=jpeg4py_loader_w_failsafe):
        """
        args:
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the VisEvent dataset.
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        root = env_settings().fe108_dir if root is None else root
        super().__init__('FE108', root, image_loader)

        self.split = split
        self.sequence_list = self._get_sequence_list()
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'fe108_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'fe108_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]
        else:
            seq_names = self.sequence_list
        self.sequence_list = [i for i in seq_names]

    def _get_sequence_list(self):
        seq_list = os.listdir(self.root)
        return seq_list

    def get_name(self):
        return 'fe108'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)


    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)  # xywh just one kind label
        # if the box is too small, it will be ignored
        valid = (bbox[:, 2] > 1.0) & (bbox[:, 3] > 1.0)
        visible = valid
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_data_path(self, seq_path, frame_id):
        ''' return img+event path '''
        img_path = os.path.join(seq_path, 'img','{:04}.jpg'.format(frame_id+1))
        event_path = os.path.join(seq_path, 'evimg','{:04}.jpg'.format(frame_id+1))
        return img_path, event_path

    def _get_frame(self, seq_path, frame_id):
        ''' Return:img+event frame '''
        img_path, event_path = self._get_data_path(seq_path, frame_id)
        # [H,W,6]:[H,W,3]+[H,W,3]
        img = get_merge_frame(img_path, event_path)
        return img

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
