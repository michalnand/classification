import torch
import torch.nn as nn

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_encoder_0 = [ 
                        self.conv_bn(input_shape[0], 32, 2),

                        self.conv_dw(32, 64, 1),
                        self.conv_dw(64, 128, 2),

                        self.conv_dw(128, 128, 1),
                        self.conv_dw(128, 256, 2)
        ]

        self.layers_encoder_1 = [
                        self.conv_dw(256, 256, 1),
                        self.conv_dw(256, 256, 2),

                        self.conv_dw(256, 256, 1),
                        self.conv_dw(256, 256, 2)
        ]

        self.layers_decoder = [
            self.conv_dw(256 + 512, 256, 1), 
            self.conv_bn(256, 128, 1),
            self.conv_bn(128, 64, 1),

            nn.Conv2d(64, output_shape[0], kernel_size = 1, stride = 1, padding = 0),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
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

    def conv_bn(self, inputs, outputs, stride):
        return nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size = 3, stride = stride, padding = 1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True))

    def conv_dw(self, inputs, outputs, stride):
        return nn.Sequential(
                # dw
                nn.Conv2d(inputs, inputs, kernel_size = 3, stride = stride, padding = 1, groups=inputs),
                nn.BatchNorm2d(inputs),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inputs, outputs, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True))
    
    def tconv_bn(self, inputs, outputs, stride):
        return nn.Sequential(
                nn.ConvTranspose2d(inputs, outputs, kernel_size = 3, stride = stride, padding = 1, output_padding = 1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True))

                

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

    print(x.shape, y.shape)




