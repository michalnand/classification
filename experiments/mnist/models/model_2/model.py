import torch
import torch.nn as nn
import numpy


class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.Conv2d(input_shape[0], 16, kernel_size = 3, stride = 2, padding = 0),
                        nn.ReLU(),  

                        nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 0),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 0),
                        nn.ReLU(),

                        nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 0),
                        nn.ReLU(), 

                        nn.Conv2d(128, output_shape[0], kernel_size = 1, stride = 1, padding = 0),

                        nn.AvgPool2d(3),
                        nn.Flatten()
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