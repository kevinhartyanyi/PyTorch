import torch.nn as nn
import time
import preprocess as pre
import model
import utils


category_lines, all_categories, n_letters, all_letters = pre.get_data()


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)


rnn = model.RNN(n_letters, 128, n_letters, len(all_categories))

criterion = nn.NLLLoss()
learning_rate = 0.0005

epoch = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for it in range(1, epoch + 1):
    output, loss = train(*utils.randomTrainingExample(all_categories, category_lines, n_letters, all_letters))
    total_loss += loss

    if it % print_every == 0:
        print('%s (%d %d%%) %.4f' % (utils.timeSince(start), it, it / epoch * 100, loss))
        torch.save(rnn.state_dict(), "basic.pth")

    if it % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

utils.plot(all_losses)

torch.save(rnn.state_dict(), "basic.pth")
