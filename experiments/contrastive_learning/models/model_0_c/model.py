import torch
import torch.nn as nn
import numpy


class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_features = [ 
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

        self.layers_output = [
            nn.Linear(128, output_shape[0])
        ]

       
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)


        for i in range(len(self.layers_output)):
            if hasattr(self.layers_output[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_output[i].weight)
        
        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        path = "./models/model_0/trained/"
        self.model_features.load_state_dict(torch.load(path   + "model.pt", map_location = self.device))
        self.model_features.eval()

        print(self.model_features)
        print(self.model_output)

    def forward(self, x):
        f = self.model_features(x).detach()
        return self.model_output(f)
    
    def save(self, path):
        torch.save(self.model_features.state_dict(), path   + "model_features.pt")
        torch.save(self.model_output.state_dict(), path   + "model_output.pt")

    def load(self, path):
        self.model_features.load_state_dict(torch.load(path   + "model_features.pt", map_location = self.device))
        self.model_features.eval() 

        self.model_output.load_state_dict(torch.load(path   + "model_output.pt", map_location = self.device))
        self.model_output.eval() 
