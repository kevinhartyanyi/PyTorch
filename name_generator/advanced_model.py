import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.fc = nn.Linear(n_categories + input_size, hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input), 1)
        inp = self.relu(self.fc(input_combined)).view(1,1,-1)
        hidden = hidden.view(1,1,-1)
        output, hidden = self.gru(inp, hidden)
        output = self.fc2(output.view(1,-1))
        output = self.softmax(output).view(1,-1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

