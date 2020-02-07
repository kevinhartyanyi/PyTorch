import preprocess as pre
import model
import utils
import torch

category_lines, all_categories, n_letters, all_letters = pre.get_data()
rnn = model.RNN(n_letters, 128, n_letters, len(all_categories))

rnn.load_state_dict(torch.load("advacnced_start_optim.pth"))

#name = utils.sample("Spanish", all_categories, n_letters, all_letters, rnn, start="AB", max_length=20)

#print(name)


utils.samples('Russian', all_categories, n_letters, all_letters, rnn, 'RUS', start_token=True)
utils.samples('German', all_categories, n_letters, all_letters, rnn, 'GER', start_token=True)
utils.samples('Spanish', all_categories, n_letters, all_letters, rnn, 'SPA', start_token=True)
utils.samples('Chinese', all_categories, n_letters, all_letters, rnn, 'CHI', start_token=True)