import argparse
import os

import editdistance as editdistance
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)


from src.datautils.batch_collate import CollateImageLabelMultiDecodersV3
from src.datautils.bressay_dataset import BressayDataset
from src.datautils.generic_charset import GenericCharset
from src.datautils.token_dataset_bressay import TOKEN_POSITION_CHAR_LEVEL_DICT, TOKEN_IS_READABLE_CHAR_LEVEL_DICT, \
    TOKEN_IS_CROSS_CHAR_LEVEL_DICT, SOS_STR_TOKEN, PAD_STR_TOKEN, TOKEN_POSITION_DICT, TOKEN_IS_READABLE_DICT, \
    TOKEN_IS_CROSS_DICT
from src.datautils.txt_transform import best_path
from src.models.models_utils import load_pretrained_model
from src.models.seq2seq import Seq2SeqMultidecoders

parser = argparse.ArgumentParser()

parser.add_argument("--dir_data", type=str, help='Path to Bressay data ex: /test')
# parser.add_argument("--dir_save", type=str)
#parser.add_argument('--level', choices=['lines', 'pages', 'paragraphs'], default='lines', required=True,
#                    help='Granularity of text level')
#parser.add_argument('--partition', choices=['training', 'validation', 'test'], default='test',
#                    help='Dataset partition to read')

# For GPU
parser.add_argument('--num_workers', default=0, type=int)

args = parser.parse_args()

# os.makedirs(args.dir_save, exist_ok=True)

height_max = 32
width_max = 768

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths
path_model = "../model_weights/seq2seq_epoch_120.torch"
charset_file = "../data/charset.txt"

# Alphabet
charset = GenericCharset(charset_file, use_blank=True, use_sos=True, use_eos=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

# Data
width_divisor = 2

fixed_size_img = (height_max, width_max)
reduce_dims_factor = np.array([height_max, width_divisor])

bressay_db = BressayDataset(args.dir_data, char_dict, fixed_size_img, reduce_dims_factor)

print('Nb samples {}:'.format(len(bressay_db)))

# Pad img with black = 0
pad_dec_txt = 0  # 0 == index of blank

c_collate_fn = CollateImageLabelMultiDecodersV3(imgs_pad_value=[0],
                                                pad_enc_txt=10000,
                                                pad_dec_txt=pad_dec_txt,
                                                pad_pos_tag=TOKEN_POSITION_DICT[PAD_STR_TOKEN],
                                                pad_iscross_tag=TOKEN_IS_CROSS_DICT[PAD_STR_TOKEN],
                                                pad_isreadable=TOKEN_IS_READABLE_DICT[PAD_STR_TOKEN])


collate_fn = c_collate_fn.collate_fn

bressay_dataloader = DataLoader(bressay_db, num_workers=args.num_workers, batch_size=32, pin_memory=True,
                                collate_fn=collate_fn,
                                shuffle=False)

nb_char_all = charset.get_nb_char()  # With blank, sos, and eos

dict_dim_out = {
    "out_enc": nb_char_all - 2,
    "out_dec_txt": nb_char_all,
    "out_dec_postag": len(TOKEN_POSITION_CHAR_LEVEL_DICT.keys()),
    "out_dec_crosstag": len(TOKEN_IS_CROSS_CHAR_LEVEL_DICT.keys()),
    "out_dec_readabletag": len(TOKEN_IS_READABLE_CHAR_LEVEL_DICT.keys()),
}

# Model
cnn_cfg = [(2, 64), (4, 128), (4, 256)]
head_cfg = (256, 3)  # (hidden , num_layers)

model = Seq2SeqMultidecoders(cnn_cfg,
                             head_cfg,
                             attention_att_hidden_size=128,
                             attention_conv_prev_att_features_number=20,
                             attention_conv_prev_att_kernel_size=7,
                             dict_dim_out=dict_dim_out,
                             context_dim=128,
                             hidden_size=256,
                             decoder_embedding_dim=64,
                             device=device)

if os.path.isfile(path_model):
    load_pretrained_model(path_model, model, device)

model = model.to(device)
model.eval()

# Evaluate
cer_total = 0
nb_total_letter_gt = 0

with torch.no_grad():
    for index_batch, batch_data in enumerate(bressay_dataloader):
        ids_batch = batch_data["ids"]
        x = batch_data["imgs"].to(device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        y_gt_txt = batch_data["label_str_raw"]
        # Remove text padding
        y_gt_txt = [t.strip() for t in y_gt_txt]

        nb_item_batch = x.shape[0]

        dec_txt, dec_tag_pos, dec_tag_cross, dec_tag_readable, encoder_out = model.predict_4decoders(
            x,
            char_dict[SOS_STR_TOKEN],
            TOKEN_POSITION_CHAR_LEVEL_DICT[SOS_STR_TOKEN],
            TOKEN_IS_CROSS_CHAR_LEVEL_DICT[SOS_STR_TOKEN],
            TOKEN_IS_READABLE_CHAR_LEVEL_DICT[SOS_STR_TOKEN])

        # Decoder
        decoder_outputs_cpu = dec_txt.cpu()
        pred_dec_txt = [best_path(l, char_list, char_dict[SOS_STR_TOKEN]) for l in decoder_outputs_cpu]

        # Remove text padding
        pred_dec_txt = [t.strip() for t in pred_dec_txt]

        # Final prediction is text decoder
        final_pred = pred_dec_txt

        cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, final_pred)]

        cer_total += sum(cers)
        nb_total_letter_gt += sum([len(t) for t in y_gt_txt])

        for p, one_id, gt_txt in zip(final_pred, ids_batch, y_gt_txt):
            print("-----Ground truth:-----")
            print(gt_txt)
            print("Prediction: ")
            print(p)

            # # Save text prediction decoder
            # path_pred_txt = os.path.join(args.dir_save, one_id + ".txt")
            #
            # #  with open(path_pred_txt, 'w', encoding="utf-8") as file:
            # with open(path_pred_txt, 'w') as file:
            #     file.write(p)

cer_total /= nb_total_letter_gt

print(f"CER: {100 * cer_total:.2f}% ")

print()
print("End")

# 2- Evaluate
# evaluate_bressay(args.dir_data, args.level, args.partition, args.dir_save)

