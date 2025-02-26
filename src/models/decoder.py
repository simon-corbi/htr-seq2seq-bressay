import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple


class Decoder(nn.Module):
    def __init__(self,
                 nbr_characters: int,
                 embed_dim: int,
                 context_dim: int,
                 hidden_size: int,
                 dropout: float = 0.5):
        """
        """

        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(nbr_characters, embed_dim)

        rnn_input_size = embed_dim + context_dim

        self.rnn = nn.LSTM(rnn_input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        fc_input_size = hidden_size + context_dim

        self.fc = nn.Linear(fc_input_size, nbr_characters)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self,
                characters: Tensor,
                previous_context: Tensor,
                context: Tensor,
                hidden_state: Tuple[Tensor]
                ):
        """
        """

        # (batch_size) -> (batch_size, embed_dim)
        # print(characters)
        embeddings = self.embedding(characters)

        # (batch_size, embed_dim) + (batch_size, context_dim) -> (batch_size, 1 (seq_len), rnn_input_size)
        rnn_input = torch.cat((embeddings, previous_context), dim=-1).unsqueeze(1)

        # (batch_size, 1 (seq_len), rnn_input_size) ->
        # (batch_size, 1 (seq_len), hidden_size) , [(1, batch_size, hidden_size), (1, batch_size, hidden_size)]
        rnn_output, new_hidden_state = self.rnn(rnn_input, hidden_state)

        # (batch_size, 1 (seq_len), hidden_size) -> (batch_size, hidden_size)
        rnn_output = rnn_output.squeeze(1)
        rnn_output = self.dropout(rnn_output)

        # (batch_size, hidden_size) + (batch_size, context_dim) -> (batch_size, fc_input_size)
        fc_input = torch.cat((rnn_output, context), dim=-1)

        # (batch_size, fc_input_size) -> (batch_size, nbr_characters)
        characters_scores = self.fc(fc_input)

        return characters_scores, new_hidden_state
