import torch
import torch.nn as nn
import numpy


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock, self).__init__()

        
        self.conv0  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0   = nn.ReLU()
        self.conv1  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1   = nn.ReLU()
            
        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)


    def forward(self, x):
        y  = self.conv0(x)
        y  = self.act0(y)
        y  = self.conv1(y)
        y  = self.act1(y + x)
        
        return y


class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        '''
        self.layers = [ 
                        nn.Conv2d(input_shape[0], 16, kernel_size = 3, stride = 1, padding = 1),
                        nn.ReLU(), 
                        nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
                        nn.ReLU(), 
                        nn.Conv2d(32, output_shape[0], kernel_size = 1, stride = 1, padding = 0) 
        ]
        '''

        self.layers = [ 
                        nn.Conv2d(input_shape[0], 32, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(), 
                        ResidualBlock(32),
                        ResidualBlock(32),

                        nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),
                        ResidualBlock(64),
                        ResidualBlock(64),

                        nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),
                        ResidualBlock(128),
                        ResidualBlock(128),

                        nn.ConvTranspose2d(128, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                        nn.ReLU(),

                        nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.ReLU(),
                        
                        nn.Conv2d(64, output_shape[0], kernel_size = 1, stride = 1, padding = 0)
        ]
        
     
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)

    def forward(self, x):
        return self.model.forward(x)
    
    def save(self, path):
        name = path + "model.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name) 

    def load(self, path):
        name = path + "model.pt"
        print("loading", name)

        self.model.load_state_dict(torch.load(name, map_location = self.device))
        self.model.eval() 

if __name__ == "__main__":
    batch_size = 8

    channels    = 3
    height      = 256
    width       = 256

    classes_count = 30


    model = Create((channels, height, width), (classes_count, height, width))

    x = torch.randn((batch_size, channels, height, width))


    y = model.forward(x)

    print(y.shape)




