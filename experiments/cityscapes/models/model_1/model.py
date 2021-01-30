import torch
import torch.nn as nn
import numpy

import time

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        self.conv_bn(input_shape[0], 32, 2),

                        self.conv_dw(32, 64, 1),
                        self.conv_dw(64, 128, 2),

                        self.conv_dw(128, 128, 1),
                        self.conv_dw(128, 256, 2),

                        self.conv_dw(256, 256, 1),
                        self.conv_dw(256, 512, 2),

                        self.conv_dw(512, 512, 1),

                        self.tconv_bn(512, 256, 2),
                        self.conv_dw(256, 256, 1),

                        self.tconv_bn(256, 128, 2),
                        self.conv_dw(128, 128, 1),

                        self.tconv_bn(128, 64, 2),
                        self.conv_dw(64, 64, 1),
                        
                        nn.Conv2d(64, output_shape[0], kernel_size = 1, stride = 1, padding = 0),
                        nn.Upsample(scale_factor=2, mode='nearest')
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

    def conv_bn(self, inputs, outputs, stride):
        return nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size = 3, stride = stride, padding = 1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True)
                )

    def conv_dw(self, inputs, outputs, stride):
        return nn.Sequential(
                # dw
                nn.Conv2d(inputs, inputs, kernel_size = 3, stride = stride, padding = 1, groups=inputs),
                nn.BatchNorm2d(inputs),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inputs, outputs, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
                )
    
    def tconv_bn(self, inputs, outputs, stride):
        return nn.Sequential(
                nn.ConvTranspose2d(inputs, outputs, kernel_size = 3, stride = stride, padding = 1, output_padding = 1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True)
                )

                

if __name__ == "__main__":
    batch_size = 1

    channels    = 3
    height      = 256
    width       = 512

    classes_count = 30


    model = Create((channels, height, width), (classes_count, height, width))

    x = torch.randn((batch_size, channels, height, width))

    model.eval()
    y = model.forward(x)

    time_start = time.time()
    for i in range(10):
        y = model.forward(x)
    time_stop = time.time()

    fps = 10.0/(time_stop-time_start)

    print(x.shape, y.shape, fps)




