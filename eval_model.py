# imports
from train_q2 import LETTERS_NUM, LANGUAGES, END_TOKEN
from train_q2 import RNN, name_to_tensor
import torch
import string


def generate_name(model, letter, language):
    # entering evaluation mode
    model.eval()

    input = name_to_tensor(letter)[0]
    hidden = model.init_hidden()
    generated_name = letter

    with torch.no_grad():

        while True:
            output, hidden = model(input, hidden, language)

            # predict the next letter
            output = torch.reshape(output, (1, LETTERS_NUM))
            predicted_letter = model.predict(output)

            # add the predicated letter to the generated name
            generated_name += predicted_letter

            # END TOKEN indicated the end of the word
            if predicted_letter == END_TOKEN:
                generated_name = generated_name.replace(END_TOKEN, '')
                break

            # referred to the predicate letter as the next input
            input = name_to_tensor(predicted_letter)[0]

    return generated_name


def evaluate_model_q2(letter, language):
    rnn = RNN(input_size=LETTERS_NUM, hidden_size=500, output_size=LETTERS_NUM)
    rnn.load_state_dict(torch.load("trained_model_q2.pkl"))

    generated_name = generate_name(rnn, letter, language)

    print(f"{generated_name}\n")


def main():
    # get input from the user (letter, language, name length)

    # checking that the letter input is valid
    while True:
        letter = input(f"Please enter a capital letter: ")
        if letter in string.ascii_uppercase:
            break

        print(f"Please enter a valid letter\n")

    print()

    # checking if the language input is valid
    while True:
        language = input("please enter a language: ")
        if language in LANGUAGES:
            break

        print(f"Please enter a valid langauge:\n{LANGUAGES}\n")

    evaluate_model_q2(letter, language)

if __name__ == '__main__':
    main()