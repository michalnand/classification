import torch
import torch.nn as nn
import numpy


class Create(torch.nn.Module):

    def __init__(self, input_shape, features_count):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_featues = [ 
                        nn.Conv2d(input_shape[0], 16, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  
                        nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.ReLU(),
                       
                        nn.Flatten(),
                        nn.Linear(7*7*64, features_count),
                        nn.ReLU(),
        ]

        
        self.layers_predictor = [
            nn.Linear(features_count, features_count)
        ]
        

       
        for i in range(len(self.layers_featues)):
            if isinstance(self.layers_featues[i], nn.Conv2d):
                torch.nn.init.orthogonal_(self.layers_featues[i].weight, 0.1)
                torch.nn.init.zeros_(self.layers_featues[i].bias)

            if isinstance(self.layers_featues[i], nn.Linear):
                torch.nn.init.xavier_uniform_(self.layers_featues[i].weight)
                torch.nn.init.zeros_(self.layers_featues[i].bias)

        torch.nn.init.xavier_uniform_(self.layers_predictor[0].weight)
        torch.nn.init.zeros_(self.layers_predictor[0].bias)
        
        self.model_features = nn.Sequential(*self.layers_featues)
        self.model_features.to(self.device)

        self.model_predictor = nn.Sequential(*self.layers_predictor)
        self.model_predictor.to(self.device)

        print(self.model_features)
        print(self.model_predictor)

    #input shape : (batch, 2, channel, height, width)
    def forward(self, x):
        xt = torch.transpose(x, 0, 1)

        za = self.model_features(xt[0])
        zb = self.model_features(xt[1])
 
        ca = self.model_predictor(za)

        logits = torch.matmul(ca, zb.t())
        
        return logits

    def eval_features(self, x):
        return self.model_features(x)
    
    def save(self, path):
        torch.save(self.model_features.state_dict(), path    + "model_features.pt")
        torch.save(self.model_predictor.state_dict(), path   + "model_predictor.pt")

    def load(self, path):
        self.model_features.load_state_dict(torch.load(path   + "model_features.pt", map_location = self.device))
        self.model_predictor.load_state_dict(torch.load(path   + "model_predictor.pt", map_location = self.device))

if __name__ == "__main__":

    input_shape     = (1, 28, 28)

    model = Create(input_shape, 128)

    x     = torch.randn((32, 2) + input_shape)
    y     = model(x)

    loss = torch.nn.functional.cross_entropy(y, torch.arange(y.shape[0]).to(y.device))

    print(y.shape)
    print(loss)