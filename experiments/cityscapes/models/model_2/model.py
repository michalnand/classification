import torch
import torch.nn as nn
import numpy

import time

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock, self).__init__()

        
        self.conv0  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn0    = nn.BatchNorm2d(channels)
        self.act0   = nn.ReLU()
        self.conv1  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1    = nn.BatchNorm2d(channels)
        self.act1   = nn.ReLU()
            
        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)

    def forward(self, x):
        y   = self.conv0(x)
        y   = self.bn0(y)   
        y   = self.act0(y)
        y   = self.conv1(y)
        y   = self.bn1(y)
        y   = self.act1(y + x)
        
        return y

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_encoder_0 = [ 
            nn.Conv2d(input_shape[0], 64, kernel_size = 8, stride = 4, padding=2),
            nn.ReLU(),
            
            ResidualBlock(64),
            ResidualBlock(64)
        ]

        self.layers_encoder_1 = [ 
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding=1),
            nn.ReLU(),
            
            ResidualBlock(128),
            ResidualBlock(128),

            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding=1),
            nn.ReLU(),

            ResidualBlock(128),
            ResidualBlock(128),

            nn.Conv2d(128, 64, kernel_size = 1, stride = 1, padding=0),
            nn.ReLU()
        ]

        self.layers_decoder = [ 
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),

            ResidualBlock(64),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, output_shape[0], kernel_size = 3, stride = 1, padding=1),
        ]

     
        for i in range(len(self.layers_encoder_0)):
            if hasattr(self.layers_encoder_0[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_encoder_0[i].weight)

        for i in range(len(self.layers_encoder_1)):
            if hasattr(self.layers_encoder_1[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_encoder_1[i].weight)

        for i in range(len(self.layers_decoder)):
            if hasattr(self.layers_decoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_decoder[i].weight)

        self.model_encoder_0 = nn.Sequential(*self.layers_encoder_0)
        self.model_encoder_0.to(self.device)

        self.model_encoder_1 = nn.Sequential(*self.layers_encoder_1)
        self.model_encoder_1.to(self.device)

        self.model_decoder = nn.Sequential(*self.layers_decoder)
        self.model_decoder.to(self.device)

        print(self.model_encoder_0)
        print(self.model_encoder_1)
        print(self.model_decoder)

    def forward(self, x):
        encoder_0 = self.model_encoder_0(x)
        encoder_1 = self.model_encoder_1(encoder_0)

        encoder_1_up = torch.nn.functional.interpolate(encoder_1, scale_factor=4, mode="nearest")
        d_in      = torch.cat([encoder_0, encoder_1_up], dim=1)

        y = self.model_decoder(d_in)
        return y
    
    def save(self, path):
        torch.save(self.model_encoder_0.state_dict(), path + "model_encoder_0.pt") 
        torch.save(self.model_encoder_1.state_dict(), path + "model_encoder_1.pt") 
        torch.save(self.model_decoder.state_dict(), path + "model_decoder.pt") 

    def load(self, path):
        self.model_encoder_0.load_state_dict(torch.load(path + "model_encoder_0.pt", map_location = self.device))
        self.model_encoder_1.load_state_dict(torch.load(path + "model_encoder_1.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "model_decoder.pt", map_location = self.device))
        
        self.model_encoder_0.eval() 
        self.model_encoder_1.eval() 
        self.model_decoder.eval() 


if __name__ == "__main__":
    batch_size = 8

    channels    = 3
    height      = 480
    width       = 640

    classes_count = 5

    model = Create((channels, height, width), (classes_count, height, width))

    x = torch.randn((batch_size, channels, height, width))

    model.eval()
    y = model.forward(x)

    print(y.shape)
