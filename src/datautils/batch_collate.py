import torch
import numpy as np

from src.datautils.img_transform import pad_images_otherdim


def pad_sequences_1D(data, padding_value):
    """
    Pad data with padding_value to get same length
    """
    x_lengths = [len(x) for x in data]
    longest_x = max(x_lengths)
    padded_data = np.ones((len(data), longest_x)).astype(np.int32) * padding_value
    for i, x_len in enumerate(x_lengths):
        padded_data[i, :x_len] = data[i][:x_len]
    return padded_data


class CollateImageLabelMultiDecodersV3(object):
    """Remove label_str_raw_notag from v2"""
    def __init__(self, imgs_pad_value,
                 pad_enc_txt, pad_dec_txt, pad_pos_tag, pad_iscross_tag, pad_isreadable):  # , pad_alltag
        self.imgs_pad_value = imgs_pad_value

        self.pad_enc_txt = pad_enc_txt
        self.pad_dec_txt = pad_dec_txt

        self.pad_pos_tag = pad_pos_tag
        self.pad_iscross_tag = pad_iscross_tag
        self.pad_isreadable = pad_isreadable

        # self.pad_alltag = pad_alltag

    def collate_fn(self, batch_data):
        """
        """

        ids = [batch_data[i]["ids"] for i in range(len(batch_data))]

        label_str_raw = [batch_data[i]["labels_allformat"]["label_str_raw"] for i in range(len(batch_data))]

        label_ind_enc = [batch_data[i]["labels_allformat"]["label_ind_enc"] for i in range(len(batch_data))]
        label_ind_enc_length = [len(l) for l in label_ind_enc]
        label_ind_enc = pad_sequences_1D(label_ind_enc, padding_value=self.pad_enc_txt)
        label_ind_enc = torch.tensor(label_ind_enc).long()

        label_ind_dec = [batch_data[i]["labels_allformat"]["label_ind_dec"] for i in range(len(batch_data))]
        label_ind_dec = pad_sequences_1D(label_ind_dec, padding_value=self.pad_dec_txt)
        label_ind_dec = torch.tensor(label_ind_dec).long()

        ind_pos = [batch_data[i]["labels_allformat"]["ind_pos"] for i in range(len(batch_data))]
        ind_pos = pad_sequences_1D(ind_pos, padding_value=self.pad_pos_tag)
        ind_pos = torch.tensor(ind_pos).long()

        ind_is_cross = [batch_data[i]["labels_allformat"]["ind_is_cross"] for i in range(len(batch_data))]
        ind_is_cross = pad_sequences_1D(ind_is_cross, padding_value=self.pad_iscross_tag)
        ind_is_cross = torch.tensor(ind_is_cross).long()

        ind_is_readable = [batch_data[i]["labels_allformat"]["ind_is_readable"] for i in range(len(batch_data))]
        ind_is_readable = pad_sequences_1D(ind_is_readable, padding_value=self.pad_isreadable)
        ind_is_readable = torch.tensor(ind_is_readable).long()

        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs_shape = [batch_data[i]["img_shape"] for i in range(len(batch_data))]
        imgs_reduced_shape = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]

        imgs = pad_images_otherdim(imgs, padding_value=self.imgs_pad_value)

        imgs = torch.tensor(imgs).float()

        formatted_batch_data = {
            "ids": ids,

            "imgs": imgs,
            "imgs_shape": imgs_shape,
            "imgs_reduced_shape": imgs_reduced_shape,

            "label_str_raw": label_str_raw,

            "label_ind_enc": label_ind_enc,
            "label_ind_enc_length": label_ind_enc_length,
            "label_ind_dec": label_ind_dec,

            "ind_pos": ind_pos,
            "ind_is_cross": ind_is_cross,
            "ind_is_readable": ind_is_readable
        }

        return formatted_batch_data
