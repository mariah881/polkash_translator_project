#Script containing model components
import torch
import torch.nn as nn
import random

#Encoder
class EncoderGRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super(EncoderGRU, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        """
        src: (batch_size, seq_len) - word indices in the input sentence
        """
        embedded = self.embedding(src)  # (batch_size, seq_len, embedding_dim)
        outputs, hidden = self.gru(embedded)  # outputs: (batch_size, seq_len, hidden_dim), hidden: (num_layers, batch_size, hidden_dim)
        return outputs, hidden  # Returning both outputs and hidden

#Attention
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        """
        hidden: (num_layers, batch_size, hidden_dim) - last hidden state of the encoder
        encoder_outputs: (batch_size, seq_len, hidden_dim) - all encoder outputs
        """

        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_dim)
        attention = torch.sum(self.v * energy, dim=2)  # (batch_size, seq_len)

        attention_output = torch.softmax(attention, dim=1)
        return attention_output # (batch_size, seq_len)
    
#Decoder
class DecoderGRU(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super(DecoderGRU, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, attention):
        """
        input: (batch_size) - previous target token
        hidden: (num_layers, batch_size, hidden_dim) - previous hidden state
        encoder_outputs: (batch_size, seq_len, hidden_dim) - all encoder outputs
        attention: attention mechanism
        """
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(input))  # (batch_size, 1, embedding_dim)

        attn_weights = attention(hidden, encoder_outputs)  # (batch_size, seq_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_len)

        context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, 1, hidden_dim)
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embedding_dim + hidden_dim)

        output, hidden = self.gru(rnn_input, hidden)  # (batch_size, 1, hidden_dim)

        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)  # (batch_size, hidden_dim * 2)
        output = self.fc_out(output)  # (batch_size, output_dim)

        return output, hidden, attn_weights.squeeze(1)

#MT model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
      batch_size = src.shape[0]
      trg_len = trg.shape[1]
      trg_vocab_size = self.decoder.fc_out.out_features

      #Tensor for storing decoder outputs
      outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
      encoder_outputs, hidden = self.encoder(src)  #Encoder pass
      trg_input = trg[:, 0]

      for t in range(1, trg_len):
          #Decoder forward pass + storing its output
          output, hidden, _ = self.decoder(trg_input, hidden, encoder_outputs, self.attention)
          outputs[:, t, :] = output

          #Teacher forcing (only applied based on the specified ratio), otherwise it uses the model's prediction
          teacher_forcing_ratio = 0.5
          current_teacher_forcing_ratio = 0.5 if random.random() < teacher_forcing_ratio else 0.0
          teacher_force = random.random() < teacher_forcing_ratio
          top1 = output.argmax(1)
          trg_input = trg[:, t] if teacher_force else top1

      return outputs