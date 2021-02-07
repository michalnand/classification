import torch
import torch.nn as nn

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape, hidden_units = 128):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm    = nn.LSTM(input_size=input_shape[0], hidden_size=hidden_units, batch_first=True)
        self.dropout = nn.Dropout(p=0.01)
        self.linear  = nn.Linear(hidden_units, output_shape[0])
 
        self.lstm.to(self.device)
        self.linear.to(self.device)

        print(self.lstm)
        print(self.dropout)
        print(self.linear)

    def forward(self, x):
        #transpose to shape (batch, sequence, channel)
        xt = x.transpose(1, 2)
        
        output, (hn, cn) = self.lstm(xt)

        y = self.dropout(hn[0])

        return self.linear(y)
    
    def save(self, path):
        torch.save(self.lstm.state_dict(),      path + "model_lstm.pt") 
        torch.save(self.linear.state_dict(),    path + "model_linear.pt") 

    def load(self, path):
        self.lstm.load_state_dict(torch.load(path + "model_lstm.pt", map_location = self.device))
        self.linear.load_state_dict(torch.load(path + "model_linear.pt", map_location = self.device))

        self.lstm.eval() 
        self.linear.eval()
