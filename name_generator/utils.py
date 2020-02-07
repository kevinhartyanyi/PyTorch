import random
import time
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot(all_losses):
    plt.ylim(0,4)
    plt.plot(all_losses)
    plt.show()
    plt.ylim(0,4)
    plt.plot(all_losses)
    plt.savefig("loss_advanced_start_optim.png")

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
def inputTensor_new(line, n_letters, all_letters, add_start=True):
    length = len(line) + 1 if add_start else len(line)
    tensor = torch.zeros(length, 1, n_letters)
    if add_start:
        tensor[0][0][n_letters - 2] = 1 # START
    for li in range(len(line)):
        letter = line[li]
        if add_start:
            li = li + 1 # Shift, because of the START token
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of first letter to end (EOS) for target
def targetTensor_new(line, n_letters, all_letters):
    letter_indexes = [all_letters.find(line[li]) for li in range(len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def randomTrainingExample_new(all_categories, category_lines, n_letters, all_letters):
    category, line = randomTrainingPair(all_categories, category_lines)
    category_tensor = categoryTensor(category, all_categories)
    input_line_tensor = inputTensor_new(line, n_letters, all_letters)
    target_line_tensor = targetTensor_new(line, n_letters, all_letters)
    #print(target_line_tensor)
    return category_tensor, input_line_tensor, target_line_tensor


# Sample from a category and starting letter
def sample_new(category, all_categories, n_letters, all_letters, rnn, start="", max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category, all_categories)

        inp = inputTensor_new(start, n_letters, all_letters)
        hidden = rnn.initHidden()

        output_name = start

        if start != "": # Build up RNN
            for i in range(len(inp) - 1): # Process every letter except for the last one
                _, hidden = rnn(category_tensor, inp[i], hidden)
            inp = inp[-1].view(1,1,-1)     
        for i in range(max_length - len(start)):
            output, hidden = rnn(category_tensor, inp[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            inp = inputTensor_new(letter, n_letters, all_letters, add_start=False)

        return output_name


def evaluation(epoch, all_categories, n_letters, all_letters, rnn, start="", max_length=20, start_token=True):
    f = open("train_advanced_optim.txt", "a")
    f.write("Epoch: %s \n" % epoch)
    for category in all_categories:
        if start_token:
            name = sample_new(category, all_categories, n_letters, all_letters, rnn, start=start, max_length=max_length)
        else:
            name = sample(category, all_categories, n_letters, all_letters, rnn, start="A", max_length=max_length)
        print("%s: %s" % (category, name))        
        f.write("%s: %s \n" % (category, name))
    f.write("\n")
    f.close()



def sample(category, all_categories, n_letters, all_letters, rnn, start="A", max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category, all_categories)
        input = inputTensor(start, n_letters, all_letters)
        hidden = rnn.initHidden()

        output_name = start

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter, n_letters, all_letters)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, all_categories, n_letters, all_letters, rnn, start_letters='ABC', start_token=True):
    for start_letter in start_letters:
        if start_token:
            print(sample_new(category,all_categories, n_letters, all_letters, rnn, start_letter))
        else:
            print(sample(category,all_categories, n_letters, all_letters, rnn, start_letter))


def inputTensor(line, n_letters, all_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line, n_letters, all_letters):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def randomTrainingExample(all_categories, category_lines, n_letters, all_letters):
    category, line = randomTrainingPair(all_categories, category_lines)
    category_tensor = categoryTensor(category, all_categories)
    input_line_tensor = inputTensor(line, n_letters, all_letters)
    target_line_tensor = targetTensor(line, n_letters, all_letters)
    return category_tensor, input_line_tensor, target_line_tensor
