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

        fc_length = sequence_length//(2**5)

        self.layers = [ 
                        nn.Conv1d(channels, 8, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  #length = 256

                        nn.Dropout(p=0.05),
                        nn.Conv1d(8, 16, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  #length = 128

                        nn.Dropout(p=0.05),
                        nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  #length = 64

                        nn.Dropout(p=0.05),
                        nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  #length = 32

                        nn.Dropout(p=0.05),
                        nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  #length = 16

                        nn.Dropout(p=0.05),
                        nn.Conv1d(64, output_shape[0], kernel_size = 1, stride = 1, padding = 0),
                        nn.AvgPool1d(kernel_size=fc_length),
                        Flatten()
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