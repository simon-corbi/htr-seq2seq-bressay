import random
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from src.datautils.token_dataset_bressay import MAX_LEN_LINES_BRESSAY, TOKEN_POSITION_CHAR_LEVEL_DICT, PAD_STR_TOKEN, \
    TOKEN_IS_CROSS_CHAR_LEVEL_DICT, TOKEN_IS_READABLE_CHAR_LEVEL_DICT
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
                 device,
                 weight_class_tag_text=1.0,
                 weight_class_tag=2.0):

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

        # Losses functions
        self.ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

        self.loss_decoder_txt = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_decoder_txt = self.loss_decoder_txt.to(device)

        # https://saturncloud.io/blog/using-weights-in-crossentropyloss-and-bceloss-pytorch/
        # Reduce weight dominant class:
        weights_classes_pos = torch.ones(self.nb_out_dec_pos, device=self.device)
        weights_classes_pos *= weight_class_tag
        weights_classes_pos[TOKEN_POSITION_CHAR_LEVEL_DICT["main"]] = weight_class_tag_text

        self.loss_decoder_pos = nn.CrossEntropyLoss(ignore_index=TOKEN_POSITION_CHAR_LEVEL_DICT[PAD_STR_TOKEN],
                                                    weight=weights_classes_pos)
        self.loss_decoder_pos = self.loss_decoder_pos.to(device)

        # Reduce weight dominant class
        weights_classes_cross = torch.ones(self.nb_out_dec_cross, device=self.device)
        weights_classes_cross *= weight_class_tag
        weights_classes_cross[TOKEN_IS_CROSS_CHAR_LEVEL_DICT["no_cross"]] = weight_class_tag_text

        self.loss_decoder_cross = nn.CrossEntropyLoss(ignore_index=TOKEN_IS_CROSS_CHAR_LEVEL_DICT[PAD_STR_TOKEN],
                                                      weight=weights_classes_cross)
        self.loss_decoder_cross = self.loss_decoder_cross.to(device)

        # Reduce weight dominant class
        weights_classes_readable = torch.ones(self.nb_out_dec_readable, device=self.device)
        weights_classes_readable *= weight_class_tag
        weights_classes_readable[TOKEN_IS_READABLE_CHAR_LEVEL_DICT["readable"]] = weight_class_tag_text

        self.loss_decoder_readable = nn.CrossEntropyLoss(ignore_index=TOKEN_IS_READABLE_CHAR_LEVEL_DICT[PAD_STR_TOKEN],
                                                         weight=weights_classes_readable)
        self.loss_decoder_readable = self.loss_decoder_readable.to(device)


    def forward_conv_onedecoder(self, target, max_seq_len, enc_seq_len, encoder_out_, use_teacher_forcing,
                                p_teacher_forcing, decoder, outdecsize, attention_module):
        """
        Forward conv features to a specific decoder
        """
        # max_seq_len = target.shape[1] - 1  # Because we do not count the first <sos> token
        batch_size = target.shape[0]

        # Initialize initial states of the decoder
        previous_context = torch.zeros(batch_size, self.context_dim).to(self.device)
        hidden_hc = (torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                     torch.zeros(1, batch_size, self.hidden_size).to(self.device))

        # Initialize initial state of the attention module
        # (batch_size, enc_seq_len)
        previous_attention_score = torch.zeros(batch_size, enc_seq_len).to(self.device)

        # First characters should be all <sos>
        output = target[:, 0]

        # Initialize outputs
        decoder_out = torch.zeros(batch_size, max_seq_len, outdecsize).to(self.device)

        attention_batch = []

        # Decoder predict one char per one char
        for t in range(max_seq_len):
            hidden_state = hidden_hc[0]
            hidden_state = hidden_state.squeeze(0)

            context, attention_score = attention_module(hidden_state, encoder_out_, previous_attention_score)

            # output : (batch_size, nbr_characters)
            # hidden_state : [(1, batch_size, hidden_size), (1, batch_size, hidden_size)]
            output, hidden_hc = decoder(output, previous_context, context, hidden_hc)

            decoder_out[:, t] = output

            top1 = output.max(-1)[1]

            if use_teacher_forcing:
                teacher_force = random.random() < p_teacher_forcing

                # Copy the previous value for next step
                output = target[:, t + 1] if teacher_force else top1
            else:
                output = top1

            previous_context = context
            previous_attention_score = attention_score

            attention_batch.append(attention_score)

        return decoder_out, attention_batch

    def compute_loss_4decoders(self, encoder_out, y_enc, x_reduced_len, y_len_enc, dec_txt, y_dec, dec_tag_pos,
                               y_dec_postag, dec_tag_cross, y_dec_crosstag, dec_tag_readable, y_dec_readtag):
        # Compute the losses
        total_loss = 0
        # Encoder
        encoder_outputs_main, encoder_outputs_shortcut = encoder_out
        encoder_outputs_main = torch.nn.functional.log_softmax(encoder_outputs_main, dim=-1)

        ctc_loss_main = self.ctc_loss_fn(encoder_outputs_main, y_enc, x_reduced_len, y_len_enc)

        total_loss = ctc_loss_main * 0.5

        # Decoder
        # Text
        nbr_characters = dec_txt.size(-1)
        ce_output = dec_txt.reshape(-1, nbr_characters)
        ce_target = y_dec.to(self.device)[:, 1:].reshape(-1).long()  # We do not use the <sos> token
        loss_dec_text = self.loss_decoder_txt(ce_output, ce_target)

        total_loss += loss_dec_text * 0.5

        # Position tag
        nb_tag = dec_tag_pos.size(-1)
        ce_output = dec_tag_pos.reshape(-1, nb_tag)
        ce_target = y_dec_postag.to(self.device)[:, 1:].reshape(-1).long()  # We do not use the <sos> token

        loss_dec_pos_tag = self.loss_decoder_pos(ce_output, ce_target)

        total_loss += loss_dec_pos_tag * 0.33

        #  Cross tag
        nb_tag = dec_tag_cross.size(-1)
        ce_output = dec_tag_cross.reshape(-1, nb_tag)
        ce_target = y_dec_crosstag.to(self.device)[:, 1:].reshape(-1).long()  # We do not use the <sos> token

        loss_dec_cross_tag = self.loss_decoder_cross(ce_output, ce_target)

        total_loss += loss_dec_cross_tag * 0.33

        # Readable tag
        nb_tag = dec_tag_readable.size(-1)
        ce_output = dec_tag_readable.reshape(-1, nb_tag)
        ce_target = y_dec_readtag.to(self.device)[:, 1:].reshape(-1).long()  # We do not use the <sos> token

        loss_dec_read_tag = self.loss_decoder_readable(ce_output, ce_target)

        total_loss += loss_dec_read_tag * 0.33

        return total_loss, ctc_loss_main, loss_dec_text, loss_dec_pos_tag, loss_dec_cross_tag, loss_dec_read_tag

    def forward_4decoders(self,
                          input_images: Tensor,
                          y_enc, x_reduced_len, y_len_enc, y_dec, y_dec_postag, y_dec_crosstag, y_dec_readtag,
                          use_teacher_forcing: bool,
                          p_teacher_forcing: float = 0.9
                          ) -> Tensor:
        """
        Forward pass for 4 decoders configurations:
            one for text (without tags)
            one for tag position (over, main, sub)
            one for tag is cross
            one for tag is readable
        """

        # Main head, shortcut head
        encoder_out = self.encoder(input_images)
        encoder_out_ = encoder_out[0]

        # (seq_len, batch_size, num_characters)
        encoder_out_ = encoder_out_.transpose(0, 1)

        enc_seq_len = encoder_out_.size(1)

        max_seq_len = y_dec.shape[1] - 1  # Because we do not count the first <sos> token
        dec_txt, att_txt = self.forward_conv_onedecoder(y_dec,
                                                        max_seq_len,
                                                        enc_seq_len,
                                                        encoder_out_,
                                                        use_teacher_forcing,
                                                        p_teacher_forcing,
                                                        self.decoder_text,
                                                        self.nbr_characters_decoder,
                                                        self.attention_txt)

        max_seq_len = y_dec_postag.shape[1] - 1  # Because we do not count the first <sos> token
        dec_tag_pos, att_pos = self.forward_conv_onedecoder(y_dec_postag,
                                                            max_seq_len,
                                                            enc_seq_len,
                                                            encoder_out_,
                                                            use_teacher_forcing,
                                                            p_teacher_forcing,
                                                            self.decoder_position_tag,
                                                            self.nb_out_dec_pos,
                                                            self.attention_pos)

        max_seq_len = y_dec_crosstag.shape[1] - 1  # Because we do not count the first <sos> token
        dec_tag_cross, att_cross = self.forward_conv_onedecoder(y_dec_crosstag,
                                                                max_seq_len,
                                                                enc_seq_len,
                                                                encoder_out_,
                                                                use_teacher_forcing,
                                                                p_teacher_forcing,
                                                                self.decoder_cross_tag,
                                                                self.nb_out_dec_cross,
                                                                self.attention_cross)

        max_seq_len = y_dec_readtag.shape[1] - 1  # Because we do not count the first <sos> token
        dec_tag_readable, att_read = self.forward_conv_onedecoder(y_dec_readtag,
                                                                  max_seq_len,
                                                                  enc_seq_len,
                                                                  encoder_out_,
                                                                  use_teacher_forcing,
                                                                  p_teacher_forcing,
                                                                  self.decoder_readable_tag,
                                                                  self.nb_out_dec_readable,
                                                                  self.attention_nonreadable)
        total_loss, ctc_loss_main, loss_dec_text, loss_dec_pos_tag, loss_dec_cross_tag, loss_dec_read_tag = self.compute_loss_4decoders(
            encoder_out, y_enc, x_reduced_len, y_len_enc, dec_txt, y_dec, dec_tag_pos, y_dec_postag, dec_tag_cross,
            y_dec_crosstag, dec_tag_readable, y_dec_readtag)

        return dec_txt, dec_tag_pos, dec_tag_cross, dec_tag_readable, encoder_out, \
               total_loss, ctc_loss_main, loss_dec_text, loss_dec_pos_tag, loss_dec_cross_tag, loss_dec_read_tag

    def predict_4decoders(self, input_images: Tensor, token_sos_txt, token_sos_pos, token_sos_cross,
                          token_sos_readable) -> Tensor:
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
                                                  False,
                                                  0,
                                                  self.decoder_text,
                                                  self.nbr_characters_decoder,
                                                  self.attention_txt)

        max_seq_len = MAX_LEN_LINES_BRESSAY
        dec_tag_pos, _ = self.forward_conv_onedecoder(token_sos_pos,
                                                      max_seq_len,
                                                      enc_seq_len,
                                                      encoder_out_,
                                                      False,
                                                      0,
                                                      self.decoder_position_tag,
                                                      self.nb_out_dec_pos,
                                                      self.attention_pos)

        max_seq_len = MAX_LEN_LINES_BRESSAY
        dec_tag_cross, _ = self.forward_conv_onedecoder(token_sos_cross,
                                                        max_seq_len,
                                                        enc_seq_len,
                                                        encoder_out_,
                                                        False,
                                                        0,
                                                        self.decoder_cross_tag,
                                                        self.nb_out_dec_cross,
                                                        self.attention_cross)

        max_seq_len = MAX_LEN_LINES_BRESSAY
        dec_tag_readable, _ = self.forward_conv_onedecoder(token_sos_readable,
                                                           max_seq_len,
                                                           enc_seq_len,
                                                           encoder_out_,
                                                           False,
                                                           0,
                                                           self.decoder_readable_tag,
                                                           self.nb_out_dec_readable,
                                                           self.attention_nonreadable)

        return dec_txt, dec_tag_pos, dec_tag_cross, dec_tag_readable, encoder_out
