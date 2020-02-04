import random
import time
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.show()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair(all_categories, category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category, all_categories):
    li = all_categories.index(category)
    tensor = torch.zeros(1, len(all_categories))
    tensor[0][li] = 1
    return tensor

# One-hot matrix of START and first to last letters (not including EOS) for input
def inputTensor(line, n_letters, all_letters):
    add_start = len(line) + 1
    tensor = torch.zeros(add_start, 1, n_letters)
    tensor[0][0][n_letters - 2] = 1 # START
    for li in range(len(line)):
        letter = line[li]
        s_li = li + 1 # Shifted, because of the START token
        tensor[s_li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of first letter to end (EOS) for target
def targetTensor(line, n_letters, all_letters):
    letter_indexes = [all_letters.find(line[li]) for li in range(len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def randomTrainingExample(all_categories, category_lines, n_letters, all_letters):
    category, line = randomTrainingPair(all_categories, category_lines)
    category_tensor = categoryTensor(category, all_categories)
    input_line_tensor = inputTensor(line, n_letters, all_letters)
    target_line_tensor = targetTensor(line, n_letters, all_letters)
    return category_tensor, input_line_tensor, target_line_tensor


# Sample from a category and starting letter
def sample(category, rnn, start="", max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)

        inp = inputTensor(start)
        hidden = rnn.initHidden()

        output_name = start

        if start != "": # Build up RNN
            for i in range(len(inp) - 1): # Process every letter except for the last one
                _, hidden = rnn(category_tensor, inp[i], hidden)
                
            inp = inp[-1]         

        for i in range(max_length - len(start)):
            output, hidden = rnn(category_tensor, inp[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            inp = inputTensor(letter)

        return output_name

