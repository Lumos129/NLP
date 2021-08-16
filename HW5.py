import codecs
import os
import pickle
import sys
import random
import json
import numpy as np

try:
    from seq2seq_lstm import Seq2SeqLSTM
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from seq2seq_lstm import Seq2SeqLSTM



class HW5LSTM(object):
    def __init__(self):
        self.num_samples = 1000

        self.latent_dim = 256
        self.epochs = 20
        self.batch_size = 64

        self.seq2seq = Seq2SeqLSTM(latent_dim=self.latent_dim,
                                   validation_split=0.1, lr=1e-3,
                                   verbose=True,
                                   lowercase=False,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size)

        self.model_name = './data/hw5.h5'

        self.data_path_json = './data/translation2019zh_train.json'
        self.test_path_json = './data/translation2019zh_valid.json'

    def load_texts_json(self, file_name):
        input_texts = []
        target_texts = []
        idx = 1
        with codecs.open(file_name, mode='r', encoding='utf-8') as fp:
            line = fp.readline()
            while line:
                line = line.strip()
                if len(line) > 0:
                    line_dict = json.loads(line)
                    new_input_text = line_dict['english']
                    new_target_text = line_dict['chinese']

                    input_texts.append(self.tokenize_text(new_input_text))
                    target_texts.append(self.tokenize_text(new_target_text))
                    idx += 1

                line = fp.readline()
                if idx > self.num_samples:
                    break

        return input_texts, target_texts

    def shuffle_texts(self, *args):
        indices = list(range(len(args[0])))
        random.shuffle(indices)
        input_texts = []
        target_texts = []
        for ind in indices:
            input_texts.append(args[0][ind])
            target_texts.append(args[1][ind])
        return input_texts, target_texts

    def tokenize_text(self, src):
        tokens = []
        for cur in src.split():
            tokens += list(cur)         # to char['t', 'o', 'k', 'e', 'n', 's']
            tokens.append('<space>')

        #print(' '.join(tokens[:-1]))   # ['t o k e n s']
        return ' '.join(tokens[:-1])

    def detokenize_text(self, src):
        new_text = ''
        for cur_token in src.split():   # del ' '
            if cur_token == '<space>':
                new_text += ' '
            else:
                new_text += cur_token

        return new_text.strip()

    def load_data_json(self):
        input_texts, target_texts = self.load_texts_json(self.data_path_json)
        self.input_texts_for_training, self.target_texts_for_training = self.shuffle_texts(*(input_texts, target_texts))

    def load_test_json(self):
        input_texts, target_texts = self.load_texts_json(self.test_path_json)
        self.input_texts_for_testing, self.target_texts_for_testing = self.shuffle_texts(*(input_texts, target_texts))

    def fit_json(self):
        self.load_data_json()

        self.seq2seq.fit(self.input_texts_for_training, self.target_texts_for_training)
        with open(self.model_name, 'wb') as fm:
            pickle.dump(self.seq2seq, fm, protocol=2)

    def predict_json(self):
        self.load_test_json()

        with open(self.model_name, 'rb') as fm:
            seq2seq = pickle.load(fm)

        predicted_texts = seq2seq.predict(self.input_texts_for_testing)

        # output
        for n in range(len(predicted_texts)):
            input_text = self.input_texts_for_testing[n]
            target_text = predicted_texts[n]
            print('    ' + self.detokenize_text(input_text) + '\t' + self.detokenize_text(target_text))


if __name__ == '__main__':
    hw5 = HW5LSTM()

    hw5.fit_json()
    hw5.predict_json()

    exit(0)
