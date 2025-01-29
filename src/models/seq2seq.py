import random
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from src.datautils.token_dataset_bressay import MAX_LEN_LINES_BRESSAY
from src.models.attention import AttentionSystem, BahdanauHybridAttention
from src.models.crnn import CRNN
from src.models.decoder import Decoder


class DecoderUsed(Enum):
    TEXT = 1
    TEXT_ALLTAG = 2
    TEXT_POSITION_CROSS_READABLE = 3
    TEXT_ALLTAG_2Steps = 4
    TEXT_POSITION_CROSS_READABLE_2Steps = 5

    def __str__(self):
        return self.name


class DecoderMerging(Enum):
    BASE = 1  # use decoder tag to transform decoder text
    MIX_1 = 2  # use mix of decoder tag and decoder text  -> update if decoder txt contain one symbol of tag
    MIX_2 = 3  # use mix of decoder tag and decoder text  -> update if decoder txt contain two symbols of tag
    REFINE_BASE = 4  # Post process decoder txt, encoder txt

    def __str__(self):
        return self.name


class Seq2SeqMultidecoders(nn.Module):
    """
    Define a Seq2Seq model compose of an encoder (CRNN), a decoder (LSTM) and a Bahdanau attention
    """

    def __init__(self,
                 cnn_cfg,
                 head_cfg,
                 attention_att_hidden_size: int,
                 attention_conv_prev_att_features_number: int,
                 attention_conv_prev_att_kernel_size: int,
                 dict_dim_out,
                 context_dim: int,
                 hidden_size: int,
                 decoder_embedding_dim: int,
                 device):

        super(Seq2SeqMultidecoders, self).__init__()

        self.nbr_characters_encoder = dict_dim_out["out_enc"]
        self.nbr_characters_decoder = dict_dim_out["out_dec_txt"]
        self.nb_out_dec_pos = dict_dim_out["out_dec_postag"]
        self.nb_out_dec_cross = dict_dim_out["out_dec_crosstag"]
        self.nb_out_dec_readable = dict_dim_out["out_dec_readabletag"]

        self.context_dim = context_dim
        self.hidden_size = hidden_size

        self.device = device

        self.encoder = CRNN(cnn_cfg, head_cfg, self.nbr_characters_encoder)

        attention_fn_txt = BahdanauHybridAttention(self.nbr_characters_encoder,
                                                   hidden_size,
                                                   attention_att_hidden_size,
                                                   conv_prev_att_features_number=attention_conv_prev_att_features_number,
                                                   conv_prev_att_kernel_size=attention_conv_prev_att_kernel_size)

        self.attention_txt = AttentionSystem(attention_fn_txt, self.nbr_characters_encoder, context_dim)

        # Define of all types of decoders
        self.decoder_text = Decoder(self.nbr_characters_decoder, decoder_embedding_dim, context_dim, hidden_size)

        attention_fn_pos = BahdanauHybridAttention(self.nbr_characters_encoder,
                                                   hidden_size,
                                                   attention_att_hidden_size,
                                                   conv_prev_att_features_number=attention_conv_prev_att_features_number,
                                                   conv_prev_att_kernel_size=attention_conv_prev_att_kernel_size)

        self.attention_pos = AttentionSystem(attention_fn_pos, self.nbr_characters_encoder, context_dim)

        attention_fn_cross = BahdanauHybridAttention(self.nbr_characters_encoder,
                                                     hidden_size,
                                                     attention_att_hidden_size,
                                                     conv_prev_att_features_number=attention_conv_prev_att_features_number,
                                                     conv_prev_att_kernel_size=attention_conv_prev_att_kernel_size)

        self.attention_cross = AttentionSystem(attention_fn_cross, self.nbr_characters_encoder, context_dim)

        attention_fn_nonreadable = BahdanauHybridAttention(self.nbr_characters_encoder,
                                                           hidden_size,
                                                           attention_att_hidden_size,
                                                           conv_prev_att_features_number=attention_conv_prev_att_features_number,
                                                           conv_prev_att_kernel_size=attention_conv_prev_att_kernel_size)

        self.attention_nonreadable = AttentionSystem(attention_fn_nonreadable, self.nbr_characters_encoder, context_dim)

        self.decoder_position_tag = Decoder(self.nb_out_dec_pos, decoder_embedding_dim, context_dim, hidden_size)
        self.decoder_cross_tag = Decoder(self.nb_out_dec_cross, decoder_embedding_dim, context_dim, hidden_size)
        self.decoder_readable_tag = Decoder(self.nb_out_dec_readable, decoder_embedding_dim, context_dim, hidden_size)

        print("Initializing model weights")
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward_conv_onedecoder(self, token_sos, max_seq_len, enc_seq_len, encoder_out_,
                                decoder, outdecsize, attention_module):
        """
        Forward conv features to a specific decoder
        """
        batch_size = encoder_out_.shape[0]  #target.shape[0]

        # Initialize initial states of the decoder
        previous_context = torch.zeros(batch_size, self.context_dim).to(self.device)
        hidden_hc = (torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                     torch.zeros(1, batch_size, self.hidden_size).to(self.device))

        # Initialize initial state of the attention module
        # (batch_size, enc_seq_len)
        previous_attention_score = torch.zeros(batch_size, enc_seq_len).to(self.device)

        # First characters should be all <sos>
        # output = target[:, 0]
        output = torch.zeros(batch_size, dtype=torch.int64).to(self.device)  # y_dec[:, 0]
        output[:] = token_sos

        # Initialize outputs
        decoder_out = torch.zeros(batch_size, max_seq_len, outdecsize).to(self.device)

        attention_batch = []

        # decoder predict one char per one char
        for t in range(max_seq_len):
            hidden_state = hidden_hc[0]
            hidden_state = hidden_state.squeeze(0)

            context, attention_score = attention_module(hidden_state, encoder_out_, previous_attention_score)

            # output : (batch_size, nbr_characters)
            # hidden_state : [(1, batch_size, hidden_size), (1, batch_size, hidden_size)]
            # output, hidden_hc = self.decoder_text(output, previous_context, context, hidden_hc)
            output, hidden_hc = decoder(output, previous_context, context, hidden_hc)

            decoder_out[:, t] = output

            top1 = output.max(-1)[1]

            output = top1

            previous_context = context
            previous_attention_score = attention_score

            attention_batch.append(attention_score)

        return decoder_out, attention_batch

    def predict_4decoders(self, input_images: Tensor, token_sos_txt, token_sos_pos, token_sos_cross, token_sos_readable) -> Tensor:
        """
        """

        # Main head, shortcut head
        encoder_out = self.encoder(input_images)
        encoder_out_ = encoder_out[0]

        # (seq_len, batch_size, num_characters)
        encoder_out_ = encoder_out_.transpose(0, 1)

        enc_seq_len = encoder_out_.size(1)

        max_seq_len = MAX_LEN_LINES_BRESSAY
        dec_txt, _ = self.forward_conv_onedecoder(token_sos_txt,
                                                  max_seq_len,
                                                  enc_seq_len,
                                                  encoder_out_,
                                                  self.decoder_text,
                                                  self.nbr_characters_decoder,
                                                  self.attention_txt)

        max_seq_len = MAX_LEN_LINES_BRESSAY
        dec_tag_pos, _ = self.forward_conv_onedecoder(token_sos_pos,
                                                      max_seq_len,
                                                      enc_seq_len,
                                                      encoder_out_,
                                                      self.decoder_position_tag,
                                                      self.nb_out_dec_pos,
                                                      self.attention_pos)

        max_seq_len = MAX_LEN_LINES_BRESSAY
        dec_tag_cross, _ = self.forward_conv_onedecoder(token_sos_cross,
                                                        max_seq_len,
                                                        enc_seq_len,
                                                        encoder_out_,
                                                        self.decoder_cross_tag,
                                                        self.nb_out_dec_cross,
                                                        self.attention_cross)

        max_seq_len = MAX_LEN_LINES_BRESSAY
        dec_tag_readable, _ = self.forward_conv_onedecoder(token_sos_readable,
                                                           max_seq_len,
                                                           enc_seq_len,
                                                           encoder_out_,
                                                           self.decoder_readable_tag,
                                                           self.nb_out_dec_readable,
                                                           self.attention_nonreadable)

        return dec_txt, dec_tag_pos, dec_tag_cross, dec_tag_readable, encoder_out
