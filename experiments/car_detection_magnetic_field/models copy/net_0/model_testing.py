import torch
import torch.nn as nn

import numpy

import export.LibEmbeddedNetwork

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()
        self.model = export.LibEmbeddedNetwork.ModelInterfacePython()
        

    def forward(self, x):
        batch_size = x.shape[0]

        y    = torch.zeros((batch_size, 5))

        for b in range(batch_size):
            y[b] = self._forward(x[b])

        return y 
    
    def save(self, path):
        pass

    def load(self, path):
        pass

    def _forward(self, x):
        x_np = x.detach().to("cpu").float().numpy()
        x_np = x_np.transpose()
        x_np = x_np.flatten()

        y_np = numpy.array(self.model.forward(x_np))

        return torch.from_numpy(y_np)