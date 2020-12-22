import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        channels  = input_shape[0]
        sequence_length  = input_shape[1]

        self.layers = [ 
                    
                        Flatten(),
			nn.Linear(sequence_length*channels, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, output_shape[0])
        ]
     
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)

    def forward(self, state):
        y = self.model.forward(state)
        return y
    
    def save(self, path):
        name = path + "model.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name) 

    def load(self, path):
        name = path + "model.pt"
        print("loading", name)

        self.model.load_state_dict(torch.load(name, map_location = self.device))
        self.model.eval() 
