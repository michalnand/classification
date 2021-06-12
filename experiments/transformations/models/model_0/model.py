import torch
import torch.nn as nn

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc_size = 128*(input_shape[1]//32)*(input_shape[2]//32)

        self.layers_features = [
            self.conv_bn(input_shape[0], 32, 2),
            self.conv_bn(32, 64, 2),

            self.conv_bn(64, 64, 1),
            self.conv_bn(64, 64, 2),

            self.conv_bn(64, 128, 1),
            self.conv_bn(128, 128, 2),

            self.conv_bn(128, 128, 1),
            self.conv_bn(128, 128, 2),

            nn.Flatten(),

            nn.Linear(self.fc_size, 256),
            nn.ReLU()
        ]

        self.layers_output = [
            nn.Linear(2*256, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape[0])
        ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        for i in range(len(self.layers_output)):
            if hasattr(self.layers_output[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_output[i].weight, 0.01)
                torch.nn.init.zeros_(self.layers_output[i].bias)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        print(self.model_features)
        print(self.model_output)


    def forward(self, x):
        x0        = x[:,0]
        x1        = x[:,1]

        f0  = self.model_features(x0)
        f1  = self.model_features(x1)

        y = self.model_output(torch.cat([f0, f1], dim=1))
        return y
    
    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "model_features.pt") 
        torch.save(self.model_output.state_dict(), path + "model_output.pt") 

    def load(self, path):
        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_output.load_state_dict(torch.load(path + "model_output.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_output.eval() 

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

    model = Create((channels, height, width), (outputs_count, ))
    model.eval()

    x = torch.randn((batch_size, 2, channels, height, width)).to(model.device)
    y = model.forward(x)

    print(x.shape, y.shape)




