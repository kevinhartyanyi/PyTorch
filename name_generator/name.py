import torch
import torch.nn as nn
import time
import preprocess as pre
import model
import utils
import torch.optim as optim


category_lines, all_categories, n_letters, all_letters = pre.get_data()

START = torch.zeros(1,n_letters)
START[0][n_letters - 2] = 1
EOS = torch.zeros(1,n_letters)
EOS[0][n_letters - 1] = 1


def train(category_tensor, input_line_tensor, target_line_tensor, teacher_forcing=True):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    #rnn.zero_grad()
    optim.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        if teacher_forcing:
            inp = input_line_tensor[i]
        elif i == 0:
            inp = START
        else:
            inp = letter_tensor
        output, hidden = rnn(category_tensor, inp, hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
        if teacher_forcing is False:
            _, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 2:
                letter_tensor = START
                continue
            letter_tensor = EOS if topi == n_letters - 1 else utils.inputTensor_new(all_letters[topi], n_letters, all_letters, add_start=False)[0]
            

    loss.backward()

    #for p in rnn.parameters():
    #    p.data.add_(-learning_rate, p.grad.data)

    optim.step()

    return output, loss.item() / input_line_tensor.size(0)


rnn = model.RNN(n_letters, 128, n_letters, len(all_categories))

learning_rate = 0.0005
criterion = nn.NLLLoss()
optim = optim.Adam(rnn.parameters(), lr=learning_rate)

epoch = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

save_model = "basic_start_optim.pth"

for it in range(1, epoch + 1):
    output, loss = train(*utils.randomTrainingExample_new(all_categories, category_lines, n_letters, all_letters), teacher_forcing=True)
    total_loss += loss

    if it % print_every == 0:
        print('%s (%d %d%%) %.4f' % (utils.timeSince(start), it, it / epoch * 100, loss))
        torch.save(rnn.state_dict(), save_model)
        utils.evaluation(it, all_categories, n_letters, all_letters, rnn, start_token=True)

        #utils.samples('Russian', all_categories, n_letters, all_letters, rnn, 'RUS', start_token=True)
        #utils.samples('German', all_categories, n_letters, all_letters, rnn, 'GER', start_token=True)
        #utils.samples('Spanish', all_categories, n_letters, all_letters, rnn, 'SPA', start_token=True)
        #utils.samples('Chinese', all_categories, n_letters, all_letters, rnn, 'CHI', start_token=True)

    if it % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

utils.plot(all_losses)

torch.save(rnn.state_dict(), save_model)
