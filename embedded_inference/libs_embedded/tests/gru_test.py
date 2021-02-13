import torch

class CustomGRU:
    def __init__(self, input_size, hidden_size, weight_hh, weight_ih, bias_hh, bias_ih):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.whr, self.whz, self.whn = weight_hh.chunk(3, 0)
        self.wir, self.wiz, self.win = weight_ih.chunk(3, 0)

        self.bhr, self.bhz, self.bhn = bias_hh.chunk(3)
        self.bir, self.biz, self.bin = bias_ih.chunk(3)
        
        print(self.whr.shape, self.whz.shape, self.whn.shape)
        print(self.wir.shape, self.wiz.shape, self.win.shape)

        self.bhr = self.bhr.unsqueeze(0).transpose(0, 1)
        self.bhz = self.bhz.unsqueeze(0).transpose(0, 1)
        self.bhn = self.bhn.unsqueeze(0).transpose(0, 1)

        self.bir = self.bir.unsqueeze(0).transpose(0, 1)
        self.biz = self.biz.unsqueeze(0).transpose(0, 1)
        self.bin = self.bin.unsqueeze(0).transpose(0, 1)


    def forward(self, x):
        batch_size      = x.shape[0]
        sequence_length = x.shape[1]
        
        hidden          = torch.zeros((batch_size, self.hidden_size)).transpose(0, 1)

        for t in range(sequence_length): 
            hidden = self.forward_step(x[:, t], hidden)

        return hidden.transpose(0, 1)

    '''
    def forward_step(self, x, hidden):
        x_ = torch.transpose(x, 0, 1)

        r = self.wir.mm(x_) + self.bir + self.whr.mm(hidden) + self.bhr
        r = torch.sigmoid(r)

        z = self.wiz.mm(x_) + self.biz + self.whz.mm(hidden) + self.bhz
        z = torch.sigmoid(z)

        n = self.win.mm(x_) + self.bin + r*(self.whn.mm(hidden) + self.bhn)
        n = torch.tanh(n)

        hidden_next = (1 - z)*n + z*hidden

        return hidden_next
    '''


    def forward_step(self, x, hidden):
        batch_size      = x.shape[0]
        x_ = torch.transpose(x, 0, 1)

        hidden_next = torch.zeros((self.hidden_size, batch_size))

        for i in range(self.hidden_size):
            r = self.wir[i].unsqueeze(0).mm(x_) + self.bir[i] + self.whr[i].unsqueeze(0).mm(hidden) + self.bhr[i]
            r = torch.sigmoid(r)

            z = self.wiz[i].unsqueeze(0).mm(x_) + self.biz[i] + self.whz[i].unsqueeze(0).mm(hidden) + self.bhz[i]
            z = torch.sigmoid(z)

            n = self.win[i].unsqueeze(0).mm(x_) + self.bin[i] + r*(self.whn[i].unsqueeze(0).mm(hidden) + self.bhn[i])
            n = torch.tanh(n)

            hidden_next[i] = (1 - z)*n + z*hidden[i]


        return hidden_next

batch_size          = 1
sequence_length     = 47
input_size          = 5
hidden_size         = 128


gru_ref     = torch.nn.GRU(input_size, hidden_size, batch_first=True)

gru_test    = CustomGRU(input_size, hidden_size, gru_ref.weight_hh_l0, gru_ref.weight_ih_l0, gru_ref.bias_hh_l0, gru_ref.bias_ih_l0)


x = 10*torch.randn((batch_size, sequence_length, input_size))

_, y_ref   = gru_ref(x)
y_test  = gru_test.forward(x)


error = ((y_ref - y_test)**2).mean()

print(error)