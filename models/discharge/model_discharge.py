import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)
        return hidden, cell

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, hidden, cell):
        output, (hidden, cell) = self.rnn(trg, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

# Define the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1] 
        trg_dim = src.shape[2]
        outputs = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)
        hidden, cell = self.encoder(src)

        print(f"Hidden shape from encoder is {hidden.shape}")
        print(f"Cell shape from encoder is {cell.shape} ")
        
        # Initialize the input to the decoder
        input = src[:, -1, 0:1].unsqueeze(1)  # initial input is the last element of the source sequence
        print(input.shape)
        
        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = trg is not None and torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t, :].unsqueeze(1) if teacher_force else output
        return outputs

def create_model(config, device):
    input_dim_encoder = config['model']['input_dim_encoder']
    input_dim_decoder = config['model']['input_dim_decoder']
    hidden_dim_encoder = config['model']['hidden_dim_encoder']
    hidden_dim_decoder = config['model']['hidden_dim_decoder']
    output_dim = config['model']['output_dim']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']

    enc = Encoder(input_dim_encoder, hidden_dim_encoder, num_layers, dropout)
    dec = Decoder(input_dim_decoder, output_dim, hidden_dim_decoder, num_layers, dropout)
    model = Seq2Seq(enc, dec, device).to(device)

    return model
