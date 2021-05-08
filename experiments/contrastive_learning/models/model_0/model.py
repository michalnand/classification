import torch
import torch.nn as nn
import numpy


class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.Conv2d(input_shape[0], 16, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  
                        nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.ReLU(), 
                        nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 0),
                        nn.ReLU(), 
                       
                        nn.Flatten(),
                        nn.Linear(5*5*64, 128),
                        nn.ReLU()
        ]

       
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)
        
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)

    #input shape : (batch, 2, channel, height, width)
    def forward(self, x):
        xt = torch.transpose(x, 0, 1)

        fa = self.model(xt[0])
        fb = self.model(xt[1])

        #euclidean metrics
        dist = ((fa - fb)**2).mean(dim=1)
        
        return dist
    
    def save(self, path):
        torch.save(self.model.state_dict(), path   + "model.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(path   + "model.pt", map_location = self.device))

        self.model.eval() 

if __name__ == "__main__":

    input_shape     = (1, 28, 28)
    output_shape    = (10, )

    model = Create(input_shape, output_shape)

    x     = torch.randn((32, 2) + input_shape)
    y     = model(x)

    print(y.shape)
    print(y)