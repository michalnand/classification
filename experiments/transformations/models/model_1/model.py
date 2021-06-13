import torch
import torch.nn as nn

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc_size = 128*(input_shape[2]//32)*(input_shape[3]//32)

        self.layers = [
            self.conv_bn(2*input_shape[1], 32, 2),
            self.conv_bn(32, 64, 2),

            self.conv_bn(64, 64, 1),
            self.conv_bn(64, 64, 2),

            self.conv_bn(64, 128, 1),
            self.conv_bn(128, 128, 2),

            self.conv_bn(128, 128, 1),
            self.conv_bn(128, 128, 2),

            nn.Flatten(),

            nn.Dropout(0.2),

            nn.Linear(self.fc_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape[0])
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)


    def forward(self, x):
        x_in    = x.reshape((x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
        y       = self.model(x_in)
        return y
    
    def save(self, path):
        torch.save(self.model.state_dict(), path + "model.pt") 

    def load(self, path):
        self.model.load_state_dict(torch.load(path + "model.pt", map_location = self.device))        
        self.model.eval() 

    def conv_bn(self, inputs, outputs, stride):
        return nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size = 3, stride = stride, padding = 1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True))
    
  
                

if __name__ == "__main__":
    batch_size      = 32

    channels        = 3
    height          = 256
    width           = 256

    outputs_count   = 5

    model = Create((2, channels, height, width), (outputs_count, ))
    model.eval()

    x = torch.randn((batch_size, 2, channels, height, width)).to(model.device)
    y = model.forward(x)

    print(x.shape, y.shape)




