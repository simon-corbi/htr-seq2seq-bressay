import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader


from src.datautils.batch_collate import CollateImageLabelMultiDecodersV3
from src.datautils.bressay_dataset import BressayDataset
from src.datautils.generic_charset import GenericCharset
from src.datautils.token_dataset_bressay import TOKEN_POSITION_CHAR_LEVEL_DICT, TOKEN_IS_READABLE_CHAR_LEVEL_DICT, \
    TOKEN_IS_CROSS_CHAR_LEVEL_DICT, PAD_STR_TOKEN, TOKEN_POSITION_DICT, TOKEN_IS_READABLE_DICT, \
    TOKEN_IS_CROSS_DICT, EOS_STR_TOKEN
from src.evaluation.eval_one_epoch import evaluate_one_epoch
from src.models.models_utils import load_pretrained_model
from src.models.seq2seq import Seq2SeqMultidecoders

parser = argparse.ArgumentParser()

parser.add_argument("--dir_data", type=str, help='Path to Bressay data ex: /test')
parser.add_argument('--num_workers', default=0, type=int)

args = parser.parse_args()

height_max = 32
width_max = 768

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths
path_model = "model_weights/seq2seq.torch"
charset_file = "data/charset.txt"

# Alphabet
charset = GenericCharset(charset_file, use_blank=True, use_sos=True, use_eos=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

token_eos = char_dict.get(EOS_STR_TOKEN)

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

# Evaluate
dict_result = evaluate_one_epoch(bressay_dataloader,
                                 model,
                                 device,
                                 char_list,
                                 char_dict["<BLANK>"],
                                 token_eos)

dict_result["metric_final_dec"].print_values()
dict_result["metrics_main_enc"].print_values()
dict_result["metrics_dec_txt"].print_values()

dict_result["metrics_dec_pos_tag"].print_values()
dict_result["metrics_dec_cross_tag"].print_values()
dict_result["metrics_dec_read_tag"].print_values()

print("End")
