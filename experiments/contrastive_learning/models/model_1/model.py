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
                       
                        nn.Flatten(),
                        nn.Linear(7*7*64, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128)
        ]

        self.layers_predictor = [
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128)
        ]

       
        for i in range(len(self.layers_features)):
            if isinstance(self.layers_features[i], nn.Conv2d):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 0.1)
                torch.nn.init.zeros_(self.layers_features[i].bias)

            if isinstance(self.layers_features[i], nn.Linear):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)
                torch.nn.init.zeros_(self.layers_features[i].bias)
        
        for i in range(len(self.layers_predictor)):
            if isinstance(self.layers_predictor[i], nn.Linear):
                torch.nn.init.orthogonal_(self.layers_predictor[i].weight, 0.1)
                torch.nn.init.zeros_(self.layers_predictor[i].bias)
        
        self.model_features = nn.Sequential(*self.layers_features)
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

        similarity = self.cosine_similarity(za, zb)

        return 1.0 - similarity

    def eval_features(self, x):
        return self.model_features(x)

    def loss(self, x, target):
        xt = torch.transpose(x, 0, 1)

        za = self.model_features(xt[0])
        zb = self.model_features(xt[1])

        similarity = self.cosine_similarity(za, zb)
        predicted  = 1.0 - similarity

        norm_a = (za**2).sum(dim=1)**0.5
        norm_b = (zb**2).sum(dim=1)**0.5 

        result = ((target - predicted)**2).mean()
        result+= ((1.0 - norm_a)**2).mean()
        result+= ((1.0 - norm_b)**2).mean()

        print(norm_a.mean(), norm_b.mean())

        return predicted, result


    def cosine_similarity(self, a, b):
        norm_a = ((a**2).sum(dim=1))**0.5
        norm_b = ((b**2).sum(dim=1))**0.5

        eps    = 0.0000001*torch.ones_like(norm_a)

        dot = (a*b).sum(dim=1)

        result = dot/torch.max(norm_a*norm_b, eps)

        return result
    
    def save(self, path):
        torch.save(self.model_features.state_dict(), path   + "model_features.pt")
        torch.save(self.model_predictor.state_dict(), path   + "model_predictor.pt")

    def load(self, path):
        self.model_features.load_state_dict(torch.load(path   + "model_features.pt", map_location = self.device))
        self.model_predictor.load_state_dict(torch.load(path   + "model_predictor.pt", map_location = self.device))


if __name__ == "__main__":

    input_shape     = (1, 28, 28)
    output_shape    = (10, )

    model = Create(input_shape, output_shape)

    x     = torch.randn((32, 2) + input_shape)
    y     = model(x)

    print(y.shape)
    print(y)