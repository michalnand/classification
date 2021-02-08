import torch
import torch.nn as nn

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape, hidden_units = 64):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gru        = nn.GRU(input_size=input_shape[0], hidden_size=hidden_units, batch_first=True)
        self.dropout    = nn.Dropout(p=0.01)
        self.linear     = nn.Linear(hidden_units, output_shape[0])
 
        self.gru.to(self.device)
        self.linear.to(self.device)

        print(self.gru)
        print(self.dropout)
        print(self.linear)

    def forward(self, x):
        #transpose to shape (batch, sequence, channel)
        xt = x.transpose(1, 2)
        
        output, hn = self.gru(xt)

        y = self.dropout(hn[0])

        return self.linear(y)
    
    def save(self, path):
        torch.save(self.gru.state_dict(),      path + "model_gru.pt") 
        torch.save(self.linear.state_dict(),    path + "model_linear.pt") 

    def load(self, path):
        self.gru.load_state_dict(torch.load(path + "model_gru.pt", map_location = self.device))
        self.linear.load_state_dict(torch.load(path + "model_linear.pt", map_location = self.device))

        self.gru.eval() 
        self.linear.eval()
