import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_category):
        super().__init__()
        self.hidden_size = hidden_size

        self.i2o = nn.Linear(n_category + input_size + hidden_size, output_size)
        self.i2h = nn.Linear(n_category + input_size + hidden_size, hidden_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        
        self.drop = nn.Dropout(p=0.1)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, category, inp, hidden):
        cat = torch.cat((category, inp, hidden), 1)
        hidden = self.i2h(cat)
        output = self.i2o(cat)
        output = self.o2o(torch.cat((hidden, output), 1))
        output = self.drop(output) 
        return self.soft(output), hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)