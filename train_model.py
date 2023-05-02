from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import random

import unicodedata
import string
import glob
import io
import os


# the languages in the model
LANGUAGES = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German',
             'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish',
             'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']
LANGUAGES_NUM = 18

# the letters we allow in our model: alphabet small, capital letters + end token ('$')
END_TOKEN = '$'
LETTERS = string.ascii_letters + END_TOKEN
LETTERS_NUM = len(LETTERS)

# the sample size of each language names when training the model
SAMPLE_SIZE = 50

"""data pre-processing functions"""

# convert Unicode string to ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in LETTERS) + END_TOKEN


def find_files(path):
    return glob.glob(path)


# Read a file and split into names
# remove any characters that are not allowed in the model
def read_names(filename):
    names = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def language_to_index(language):
    return LANGUAGES.index(language)


def letter_to_index(letter):
    return LETTERS.find(letter)


# Turn a name into a (name_length x 'LETTERS_NUM') matrix, consists from one hot encoding vectors
# the ith row represents the one hot encoding of the name's ith letter
def name_to_tensor(name):
    name_tensor = torch.zeros(len(name), LETTERS_NUM)

    for i, letter in enumerate(name):
        name_tensor[i][letter_to_index(letter)] = 1

    return name_tensor


def load_data():
    # category line is a dictionary defined as follows:
    # key - language
    # value - all the names under the language
    languages_names, languages_names_tensors = {}, {}

    for filename in find_files('q2_data/names/*.txt'):
        language = os.path.splitext(os.path.basename(filename))[0]
        names = read_names(filename)

        languages_names_tensors[language] = [(name_to_tensor(name), name) for name in names]

    return languages_names_tensors

# -----------------------------------------------------------------------------------

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        i2h_dict, i2o_dict = {}, {}

        # initialize a set of learnable parameters for each language
        for language in LANGUAGES:
            i2h_dict[language] = nn.Linear(input_size + hidden_size, hidden_size)
            i2o_dict[language] = nn.Linear(input_size + hidden_size, output_size)

        self.i2h = nn.ModuleDict(i2h_dict)
        self.i2o = nn.ModuleDict(i2o_dict)

        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, letter_input, hidden_input, language):
        combined = torch.cat((letter_input, hidden_input))

        hidden = self.i2h[language](combined)
        output = self.logsoftmax(self.i2o[language](combined))

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.hidden_size)

    def predict(self, output):
        predicated_indices = torch.argmax(output, dim=1)
        res = ""

        for i in predicated_indices:
            res += LETTERS[i]

        return res


def plot_graph(loss, epochs):

    plt.plot(epochs, loss)

    plt.title("Cross Entropy Loss on the Train Dataset\n as a Function of Epochs Number")
    plt.xlabel("Epochs Number")
    plt.ylabel("Cross Entropy Loss")
    plt.show()


def train_on_sample(model, name_tensor, name, language, loss_function, optimizer):
    hidden = model.init_hidden()
    output = torch.zeros(name_tensor.size())

    # we don't need to predict the first letter since we get it
    output[0] = name_tensor[0]

    for i in range(len(name)-1):
        output_vec, hidden = model(name_tensor[i], hidden, language)
        output[i+1] = output_vec

    # the model's loss on the given name
    loss = loss_function(output, name_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, languages_names_tensors, loss_function, optimizer, epochs):
    train_loss_lst = []

    for epoch in range(epochs):
        print(f"round: {epoch}")
        total_train_loss = 0
        # for each language, sample 'SAMPLE_SIZE' different names (no replacements)
        for language, name_tensor_lst in languages_names_tensors.items():
            random_name_tensor_lst = random.sample(name_tensor_lst, SAMPLE_SIZE)

            for name_tensor, name in random_name_tensor_lst:
                loss, success = train_on_sample(model, name_tensor, name, language, loss_function, optimizer)
                total_train_loss += loss

        avg_train_loss = total_train_loss / (SAMPLE_SIZE * LANGUAGES_NUM)
        train_loss_lst.append(avg_train_loss)

    return train_loss_lst



def train_model_q2():
    languages_names_tensors = load_data()

    rnn = RNN(input_size=LETTERS_NUM, hidden_size=500, output_size=LETTERS_NUM)

    loss_function = nn.CrossEntropyLoss(reduction="sum")
    optimiser = torch.optim.Adam(rnn.parameters(), lr=0.00025)
    epochs = 100

    train_loss_lst = train(rnn, languages_names_tensors, loss_function, optimiser, epochs)
    plot_graph(train_loss_lst, epochs=[i for i in range(epochs)])

    # saving the trained model
    torch.save(rnn.state_dict(), "trained_model.pkl")


def main():
    train_model_q2()


if __name__ == '__main__':
    main()
