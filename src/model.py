import torch.nn as nn
import torch

class GRUClassif(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_layers, dropout):
        super(GRUClassif, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=d_model,
            num_layers=num_layers, 
            dropout=dropout)
        self.linear = nn.Linear(d_model, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.activation(x)
        out = self.softmax(self.linear(x))

        return out

