import torch
import torch.nn as nn

class Create(torch.nn.Module):

    def __init__(self, input_shape, output_shape, hidden_units = 128):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm    = nn.LSTM(input_size=input_shape[0], hidden_size=hidden_units, batch_first=True)
        self.linear  = nn.Linear(hidden_units, output_shape[0])
 
        self.lstm.to(self.device)
        self.linear.to(self.device)

        print(self.lstm)
        print(self.linear)

    def forward(self, x):
        #transpose to shape (batch, sequence, channel)
        xt = x.transpose(1, 2)
        
        output, (hn, cn) = self.lstm(xt)

        return self.linear(hn[0])
    
    def save(self, path):
        torch.save(self.lstm.state_dict(),      path + "model_lstm.pt") 
        torch.save(self.linear.state_dict(),    path + "model_linear.pt") 

    def load(self, path):
        self.lstm.load_state_dict(torch.load(path + "model_lstm.pt", map_location = self.device))
        self.linear.load_state_dict(torch.load(path + "model_linear.pt", map_location = self.device))

        self.lstm.eval() 
        self.linear.eval()


if __name__ == "__main__":
    batch_size   = 32
    input_shape  = (4, 512)
    output_shape = (5, )

    model = Create(input_shape, output_shape)

    x = torch.zeros((batch_size, ) + input_shape)

    y = model(x)
