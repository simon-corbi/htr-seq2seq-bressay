import glob
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset

from src.datautils.img_transform import apply_preprocessing_multiscale
from src.datautils.txt_transform import gt_transform_seq2seq_txt_multidecoders_use_tag_in_txt_char_level


class BressayDataset(Dataset):
    """
    Difference with v2:
        Tag are in char level
    """

    def __init__(self,
                 dir_data: str,
                 dictionary_label,
                 fixed_size,
                 reduce_dims_factor,
                 transforms: list = None):
        """
        """

        self.image_paths = []

        self.labels_allformat = []
        self.attention_gt = []

        self.id_item = []

        self.fixed_size = fixed_size
        self.transforms = transforms
        # self.subset = subset

        self.reduce_dims_factor = reduce_dims_factor

        # Images and label
        files_img = glob.glob(dir_data + '/**/*.png', recursive=True)

        for one_file in files_img:
            # Get id file
            split_name = os.path.split(one_file)
            split_name = split_name[1].split(sep=".")  # Filename and extension
            id_file = split_name[0]

            path_label = os.path.join(dir_data, id_file + ".txt")

            # All labels formats: for encoder, decoder text, decoders tags
            gt_dict = gt_transform_seq2seq_txt_multidecoders_use_tag_in_txt_char_level(path_label, dictionary_label)

            self.labels_allformat.append(gt_dict)

            self.id_item.append(id_file)

            path_img = os.path.join(dir_data, id_file + ".png")
            self.image_paths.append(path_img)


    def __len__(self):
        """
        Returns the number of images in the dataset
        Returns
        -------
        length: int
            number of images in the dataset
        """

        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        """
        paths_img = self.image_paths[idx]
        img = io.imread(paths_img)

        # Resize and pad
        fixe_width = self.fixed_size[1]

        labels_allformat = self.labels_allformat[idx]

        img = apply_preprocessing_multiscale(img, self.fixed_size[0], fixe_width)

        # Augmentation
        if self.transforms is not None:
            for tr in self.transforms:
                if np.random.rand() < .5:
                    img = tr(img)

        imgs_shape = img.shape
        img_reduced_shape = np.floor(imgs_shape / self.reduce_dims_factor).astype(int)

        img_tensor = torch.as_tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dim

        sample = {
            "ids": self.id_item[idx],

            "labels_allformat": labels_allformat,

            "img": img_tensor,
            "img_shape": imgs_shape,
            "img_reduced_shape": img_reduced_shape,
        }

        return sample
