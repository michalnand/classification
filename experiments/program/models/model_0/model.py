from unicodedata import bidirectional
import torch
import torch.nn as nn
import numpy


class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.projection_layer   = nn.Linear(input_shape[1], 16) 
        #self.rnn_layer          = nn.LSTM(input_size = 16, hidden_size = 128, batch_first = True)
        self.rnn_layer          = nn.GRU(input_size = 16, hidden_size = 128, batch_first = True)
        self.output_layer       = nn.Linear(128, output_shape[0])    


        torch.nn.init.orthogonal_(self.projection_layer.weight, 1.0)
        torch.nn.init.zeros_(self.projection_layer.bias)

        torch.nn.init.orthogonal_(self.output_layer.weight, 1.0)
        torch.nn.init.zeros_(self.output_layer.bias)
        

        self.projection_layer.to(self.device)
        self.rnn_layer.to(self.device)
        self.output_layer.to(self.device)

        print(self.projection_layer)
        print(self.rnn_layer)
        print(self.output_layer)

    def forward(self, x):
        y = self.projection_layer(x)

        #y, (hn, cn) = self.rnn_layer(y)

        y, hn = self.rnn_layer(y)

        y = hn[0]
        y = self.output_layer(y)

        return y
    
    def save(self, path):
        print("saving", path)
        torch.save(self.projection_layer.state_dict(), path + "projection_layer.pt") 
        torch.save(self.rnn_layer.state_dict(), path + "rnn_layer.pt") 
        torch.save(self.output_layer.state_dict(), path + "output_layer.pt") 

    def load(self, path):
        print("loading", path)

        self.projection_layer.load_state_dict(torch.load(path + "projection_layer.pt", map_location = self.device))
        self.rnn_layer.load_state_dict(torch.load(path + "rnn_layer.pt", map_location = self.device))
        self.output_layer.load_state_dict(torch.load(path + "output_layer.pt", map_location = self.device))
        


if __name__ == "__main__":
    batch_size = 17

    seq_length = 51
    channels   = 71

    input_shape = (seq_length, channels)

    output_shape = (2, )

    model = Create(input_shape, output_shape)


    x = torch.randn((batch_size, ) + input_shape)
    y = model(x)