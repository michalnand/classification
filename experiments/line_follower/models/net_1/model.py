import torch
import torch.nn as nn
import numpy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//4)*(input_shape[2]//4)

        self.layers = [ 
                        nn.Conv2d(input_shape[0], 8, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  

                        nn.Conv2d(8, 8, kernel_size = 3, stride = 2, padding = 1),
                        nn.ReLU(),  

                        Flatten(),
                        nn.Linear(fc_size*8, output_shape[0])
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


    def epoch_start(self, epoch, epoch_count):        
        if epoch < 0.2*epoch_count:
            self._reinit_lazy_kernels()

        self._add_weights_quantization()

    def _reinit_lazy_kernels(self):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                w_np = self.layers[i].weight.detach().to("cpu").numpy()

                kernels_count = w_np.shape[0]
                sizes = numpy.zeros(kernels_count)
                for kernel in range(kernels_count):
                    sizes[kernel] = numpy.linalg.norm(w_np[kernel])
                
                lazy_kernel_idx = numpy.argmin(sizes)

                #print(sizes, lazy_kernel_idx)

                noise = 0.1*torch.randn(self.layers[i].weight[lazy_kernel_idx].shape)

                self.layers[i].weight[lazy_kernel_idx].data+= noise

    def _add_weights_quantization(self, bits = 8):

        bits_ = bits-1
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                
                w_np = self.layers[i].weight.detach().to("cpu").numpy()
                
                w_np = numpy.clip(numpy.ceil(w_np*(2**bits_)), -(2**bits_), (2**bits_))
                w_np = w_np*1.0/(2**bits_)

                self.layers[i].weight.data = torch.from_numpy(w_np).to(self.device)