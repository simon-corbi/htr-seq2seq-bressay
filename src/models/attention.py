import torch
import torch.nn as nn
from torch import Tensor


# Attention function
class BahdanauHybridAttention(nn.Module):
    def __init__(self,
                 nbr_characters: int = 80,
                 hidden_size: int = 256,
                 att_hidden_size=128,
                 conv_prev_att_features_number: int = 20,
                 conv_prev_att_kernel_size: int = 7):

        super(BahdanauHybridAttention, self).__init__()

        self.fc_encoder = nn.Linear(nbr_characters, att_hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, att_hidden_size)
        self.conv_prev_att = nn.Conv1d(1, conv_prev_att_features_number,
                                       kernel_size=conv_prev_att_kernel_size,
                                       padding=conv_prev_att_kernel_size // 2)
        self.fc_prev_att = nn.Linear(conv_prev_att_features_number, att_hidden_size)
        self.fc_attention = nn.Linear(att_hidden_size, 1)

    def forward(self,
                hidden_state: Tensor,
                encoder_outputs: Tensor,
                previous_attention: Tensor):
        """

        Parameters
        ----------
        hidden_state: Tensor
            Tensor of shape (batch_size, hidden_size)
        encoder_outputs: Tensor
            Tensor of shape (batch_size, enc_seq_len, nbr_chars)
        previous_attention: Tensor
            Tensor of shape (batch_size, enc_seq_len)

        Returns
        -------
        attention_probabilities: Tensor of shape (batch_size, enc_seq_len)
        """

        # (batch_size, enc_seq_len, nbr_chars) -> (batch_size, enc_seq_len, att_hidden_size)
        enc_scores = self.fc_encoder(encoder_outputs)

        # (batch_size, hidden_size) -> (batch_size, att_hidden_size)
        hidden_scores = self.fc_hidden(hidden_state)
        # (batch_size, context_dim) -> (batch_size, 1, att_hidden_size)
        hidden_scores = hidden_scores.unsqueeze(1)

        # (batch_size, enc_seq_len) -> (batch_size, 1, enc_seq_len)
        previous_attention = previous_attention.unsqueeze(1)
        # (batch_size, 1, enc_seq_len) - > (batch_size, attention_score_conv_feature_number, enc_seq_len)
        prev_att_features = self.conv_prev_att(previous_attention)
        prev_att_features = nn.functional.leaky_relu(prev_att_features)
        # (batch_size, attention_score_conv_feature_number, enc_seq_len) ->
        # (batch_size, enc_seq_len, attention_score_conv_feature_number)
        prev_att_features = prev_att_features.transpose(1, 2)
        # (batch_size, enc_seq_len, attention_score_conv_feature_number) -> (batch_size, enc_seq_len, att_hidden_size)
        prev_att_scores = self.fc_prev_att(prev_att_features)

        # (batch_size, enc_seq_len, att_hidden_size) + (batch_size, 1, att_hidden_size)
        # + (batch_size, enc_seq_len, att_hidden_size)
        # -> (batch_size, enc_seq_len, att_hidden_size)
        combined_scores = enc_scores + hidden_scores + prev_att_scores
        combined_scores = torch.tanh(combined_scores)

        # (batch_size, enc_seq_len, att_hidden_size) -> (batch_size, enc_seq_len, 1)
        attention_scores = self.fc_attention(combined_scores)
        # (batch_size, enc_seq_len, 1) -> (batch_size, enc_seq_len)
        attention_scores = attention_scores.squeeze(2)

        attention_probabilities = torch.softmax(attention_scores, dim=-1)

        return attention_probabilities


# Attention mecanism
class AttentionSystem(nn.Module):
    def __init__(self,
                 attention_function: nn.Module,
                 nbr_characters: int,
                 context_dim: int):
        super(AttentionSystem, self).__init__()

        self.attention = attention_function
        self.fc = nn.Linear(nbr_characters, context_dim)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self,
                hidden_state: Tensor,
                encoder_outputs: Tensor,
                previous_attention: Tensor):
        """

        Parameters
        ----------
        hidden_state: Tensor
            Tensor of shape (batch_size, hidden_size)
        encoder_outputs: Tensor
            Tensor of shape: (batch_size, enc_seq_len, nbr_characters)
        previous_attention: Tensor
            Tensor of shape (batch_size, enc_seq_len)

        Returns
        -------
        context: Tensor
            Tensor of shape (batch_size, context_dim)
        """

        # (batch_size, enc_seq_len)
        attention_scores = self.attention(hidden_state=hidden_state,
                                          encoder_outputs=encoder_outputs,
                                          previous_attention=previous_attention)

        # (batch_size, enc_seq_len) -> (batch_size, 1, enc_seq_len)
        attention_scores_ = attention_scores.unsqueeze(1)

        # (batch_size, 1, enc_seq_len) x (batch_size, enc_seq_len, nbr_characters)
        # -> (batch_size, 1, nbr_characters)
        context = torch.matmul(attention_scores_, encoder_outputs)
        # (batch_size, 1, nbr_characters) -> (batch_size, nbr_characters)
        context = context.squeeze(1)

        # (batch_size, nbr_characters) -> (batch_size, context_dim)
        context = self.fc(context)
        context = torch.sigmoid(context)

        return context, attention_scores

