from unicodedata import bidirectional
import torch
import torch.nn as nn
import numpy


class ResidualLayer(torch.nn.Module):
    def __init__(self, features_count):
        super(ResidualLayer, self).__init__()

        self.lin_0 = torch.nn.Linear(features_count, features_count)
        self.lin_1 = torch.nn.Linear(features_count, features_count)

        '''
        torch.nn.init.orthogonal_(self.lin_0.weight, 2.0**0.5)
        torch.nn.init.orthogonal_(self.lin_1.weight, 2.0**0.5)
        '''

        torch.nn.init.zeros_(self.lin_0.bias)
        torch.nn.init.zeros_(self.lin_1.bias)


    def forward(self, x):

        y = self.lin_0(x)
        y = torch.relu(y)

        y = self.lin_1(y)
        y = torch.relu(y + x)

        return y


class AttentionLayer(torch.nn.Module):
    def __init__(self, features_count):
        super(AttentionLayer, self).__init__()

        self.wq = torch.nn.Linear(features_count, features_count)
        self.wk = torch.nn.Linear(features_count, features_count)
        self.wv = torch.nn.Linear(features_count, features_count)

        '''
        torch.nn.init.orthogonal_(self.wq.weight, 1.0)
        torch.nn.init.orthogonal_(self.wk.weight, 1.0)
        torch.nn.init.orthogonal_(self.wv.weight, 1.0)
        '''

        torch.nn.init.zeros_(self.wq.bias)
        torch.nn.init.zeros_(self.wk.bias)
        torch.nn.init.zeros_(self.wv.bias) 


    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        attn = torch.bmm(q, torch.transpose(k, 1, 2))
        attn = torch.softmax(attn/(q.shape[2]**0.5), dim=2)

        y    = torch.bmm(attn, v)

        return y + x

class TransformerLayer(torch.nn.Module):
    def __init__(self, features_count):
        super(TransformerLayer, self).__init__()

        self.attention = AttentionLayer(features_count)
        self.residual  = ResidualLayer(features_count)
        

    def forward(self, x):
        x = self.attention(x)
        x = self.residual(x)
    
        return x


class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        features_count = 64
        depth          = 4

        self.projection_layer   = nn.Linear(input_shape[1], features_count) 

        t_layers = []
        for _ in range(depth):
            t_layers.append(TransformerLayer(features_count))

        self.transformer_layer = torch.nn.Sequential(*t_layers)

        self.output_layer = nn.Linear(features_count, output_shape[0])


        self.projection_layer.to(self.device)
        self.transformer_layer.to(self.device)
        self.output_layer.to(self.device)

        print(self.projection_layer)
        print(self.transformer_layer)
        print(self.output_layer)

    def forward(self, x):
        y = self.projection_layer(x)
        y = self.transformer_layer(y)

        y = torch.mean(y, dim = 1)
        y = self.output_layer(y)

        return y
    
    def save(self, path):
        print("saving", path)
        torch.save(self.projection_layer.state_dict(), path + "projection_layer.pt") 
        torch.save(self.transformer_layer.state_dict(), path + "transformer_layer.pt") 
        torch.save(self.output_layer.state_dict(), path + "output_layer.pt") 

    def load(self, path):
        print("loading", path)

        self.projection_layer.load_state_dict(torch.load(path + "projection_layer.pt", map_location = self.device))
        self.transformer_layer.load_state_dict(torch.load(path + "transformer_layer.pt", map_location = self.device))
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
    
    print(">>>> ", y.shape)