import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datautils.batch_collate import CollateImageLabelMultiDecodersV3
from src.datautils.bressay_dataset import BressayDataset
from src.datautils.generic_charset import GenericCharset
from src.datautils.img_transform import affine_transformation
from src.datautils.token_dataset_bressay import TOKEN_POSITION_CHAR_LEVEL_DICT, TOKEN_IS_READABLE_CHAR_LEVEL_DICT, \
    TOKEN_IS_CROSS_CHAR_LEVEL_DICT, SOS_STR_TOKEN, PAD_STR_TOKEN, TOKEN_POSITION_DICT, TOKEN_IS_READABLE_DICT, \
    TOKEN_IS_CROSS_DICT, EOS_STR_TOKEN
from src.evaluation.eval_one_epoch import evaluate_one_epoch
from src.models.models_utils import load_pretrained_model
from src.models.seq2seq import Seq2SeqMultidecoders
from src.train.train_one_epoch import train_one_epoch

parser = argparse.ArgumentParser()

parser.add_argument("dataset_folder")
parser.add_argument("log_dir")

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--debug_pc', default=0, type=int)
parser.add_argument('--nb_epochs_max', default=2, type=int)
parser.add_argument("--path_model", default="", help="path of pretrained model", type=str)
# parser.add_argument('--height_max', default=0, type=int)
# parser.add_argument('--width_max', default=0, type=int)

parser.add_argument('--milestones_1', default=40, type=int)
parser.add_argument('--milestones_2', default=80, type=int)

parser.add_argument('--weight_class_tag_text', default=1.0, type=float)
parser.add_argument('--weight_class_tag', default=2.0, type=float)

args = parser.parse_args()

height_max = 32
width_max = 768

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device :")
print(device)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))

# Paths
charset_file = "data/charset.txt"

directory_log = args.log_dir

directory_train = os.path.join(args.dataset_folder, "train")
directory_val = os.path.join(args.dataset_folder, "validation")
directory_test = os.path.join(args.dataset_folder, "test")

# Alphabet
charset = GenericCharset(charset_file, use_blank=True, use_sos=True, use_eos=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

# Data
width_divisor = 2

fixed_size_img = (height_max, width_max)
reduce_dims_factor = np.array([height_max, width_divisor])

aug_transforms = [lambda x: affine_transformation(x, s=.1)]

train_db = BressayDataset(directory_train, char_dict, fixed_size_img, reduce_dims_factor,
                          transforms=aug_transforms)
val_db = BressayDataset(directory_val, char_dict, fixed_size_img, reduce_dims_factor)
test_db = BressayDataset(directory_test, char_dict, fixed_size_img, reduce_dims_factor)

print('Nb samples train {}:'.format(len(train_db)))
print('Nb samples val {}:'.format(len(val_db)))
print('Nb samples test {}:'.format(len(test_db)))

token_sos = char_dict.get(SOS_STR_TOKEN)
token_eos = char_dict.get(EOS_STR_TOKEN)

# Pad img with black = 0
pad_dec_txt = 0  # 0 == index of blank

c_collate_fn = CollateImageLabelMultiDecodersV3(imgs_pad_value=[0],
                                                pad_enc_txt=10000,
                                                pad_dec_txt=pad_dec_txt,
                                                pad_pos_tag=TOKEN_POSITION_DICT[PAD_STR_TOKEN],
                                                pad_iscross_tag=TOKEN_IS_CROSS_DICT[PAD_STR_TOKEN],
                                                pad_isreadable=TOKEN_IS_READABLE_DICT[PAD_STR_TOKEN])

collate_fn = c_collate_fn.collate_fn

train_dataloader = DataLoader(train_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                              collate_fn=collate_fn, shuffle=True)

val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                            collate_fn=collate_fn, shuffle=False)

test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                             collate_fn=collate_fn, shuffle=False)

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

model_reco = Seq2SeqMultidecoders(cnn_cfg,
                                  head_cfg,
                                  attention_att_hidden_size=128,
                                  attention_conv_prev_att_features_number=20,
                                  attention_conv_prev_att_kernel_size=7,
                                  dict_dim_out=dict_dim_out,
                                  context_dim=128,
                                  hidden_size=256,
                                  decoder_embedding_dim=64,
                                  device=device)

print(f"Transferring model to {str(device)}...")
model_reco = model_reco.to(device)

number_parameters = sum(p.numel() for p in model_reco.parameters() if p.requires_grad)
print(f"Model has {number_parameters:,} trainable parameters.")

optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

best_cer = 1.0
best_epoch = 0

p_teacher_forcing = 1.0

path_save_model_best = os.path.join(directory_log, "seq2seq_best.torch")
path_save_model_last = os.path.join(directory_log, "seq2seq_last.torch")

lr = args.learning_rate

begin_train = time.time()

for epoch in range(0, args.nb_epochs_max):
    begin_time_epoch = time.time()
    print('EPOCH {}:'.format(epoch))

    # Learning rate values
    if epoch < args.milestones_1:
        lr = args.learning_rate
    elif epoch < args.milestones_2:
        lr = args.learning_rate / 10
    else:
        lr = args.learning_rate / 10

    for g in optimizer.param_groups:
        g['lr'] = lr
    print("lr:" + str(lr))

    # Teacher forcing decoder(s)
    if epoch < args.milestones_1:
        p_teacher_forcing = 1.0
    elif epoch < args.milestones_2:
        p_teacher_forcing = 0.9
    else:
        p_teacher_forcing = 0.8

    # Training
    dict_losses = train_one_epoch(train_dataloader,
                                  optimizer,
                                  model_reco,
                                  device,
                                  p_teacher_forcing)

    print('trsin_total_loss_e {}'.format(dict_losses["total_loss_e"]))
    print('train_loss_enc_main {}'.format(dict_losses["loss_enc_main_epoch"]))
    print('train_loss_dec_text_epoch {}'.format(dict_losses["loss_dec_text_epoch"]))

    print('train_loss_dec_pos_epochtrain_ {}'.format(dict_losses["loss_dec_pos_epoch"]))
    print('train_loss_dec_readable_epoch {}'.format(dict_losses["loss_dec_readable_epoch"]))

    # Evaluate ALL
    print("--------------Evaluate all-------------------------------")
    dict_result = evaluate_one_epoch(val_dataloader,
                                     model_reco,
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

    # Save model
    if dict_result["metrics_dec_txt"].cer < best_cer:
        best_cer = dict_result["metrics_dec_txt"].cer
        best_epoch = epoch
        print("Best cer final, save model.")
        torch.save(model_reco.state_dict(), path_save_model_best)

    end_time_epoch = time.time()
    print("Time one epoch (s): " + str((end_time_epoch - begin_time_epoch)))
    print("")

    torch.save(model_reco.state_dict(), path_save_model_last)

end_train = time.time()
print("best_epoch: " + str(best_epoch))
print("best_cer val: " + str(best_cer))
print("Time all (s): " + str((end_train - begin_train)))
print("End training")

print("--------Begin Testing best-----------")
# Load best model
if os.path.isfile(path_save_model_best):
    load_pretrained_model(path_save_model_best, model_reco, device)

dict_result = evaluate_one_epoch(test_dataloader,
                                 model_reco,
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

print()
print("End")
