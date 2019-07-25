from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import KeyedVectors

from collections import defaultdict
import time
import random
import numpy as np
import json

from typing import Tuple, Dict


class Corpus():
    """
    Represents the training corpus.
    
    training_data: A list of tuples of the form (list of words, label)
    dev_data: A list of tuples of the form (list of words, label)
    
    w2i: A word to index map
    t2i: A label to index map
    """
    def __init__(self, training_data: Tuple[list, int],
                        dev_data: Tuple[list, int],
                        test_data: Tuple[list, int],
                        w2i: Dict[str, int],
                        t2i: Dict[str, int]):
        sorted_training_data = sorted(training_data, key=lambda sentence_pair: len(sentence_pair[0]))
        self.training_input = [pair[0] for pair in sorted_training_data]
        self.training_labels = [pair[1] for pair in sorted_training_data]

        self.dev_input = [pair[0] for pair in dev_data]
        self.dev_labels = [pair[1] for pair in dev_data]

        self.test_input = [pair[0] for pair in test_data]
        self.test_labels = [pair[1] for pair in test_data]

        self.w2i = w2i
        self.t2i = t2i

        self.i2w = {}
        for word, ind in self.w2i.items():
            self.i2w[ind] = word

        self.i2t = {}
        for label, ind in self.t2i.items():
            self.i2t[ind] = label

# Functions to read in the corpus
def read_dataset(filename: str, w2i: dict, t2i: dict, to_lower: bool):
    with open(filename, "r", encoding='utf-8') as f:
        for line in f:
            tag, words = line.strip().split(" ||| ")
            if to_lower:
                words = words.lower()
            yield ([w2i[x] for x in words.split()], t2i[tag])

def load_data(train_file: str, dev_file: str, test_file: str, to_lower: bool):
    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    UNK = w2i["<unk>"]
    
    training_data = list(read_dataset(train_file, w2i, t2i, to_lower))
    
    # If a word is not found in the test data, we'll consider it to be UNK.
    w2i = defaultdict(lambda: UNK, w2i)
    dev_data = list(read_dataset(dev_file, w2i, t2i, to_lower))

    test_data = list(read_dataset(test_file, w2i, t2i, to_lower))
    if 'unk' in t2i:
        del t2i['unk']
    if 'UNK' in t2i:
        del t2i['UNK']

    return Corpus(training_data, dev_data, test_data, w2i, t2i)

def word2vec_populate_pretrained_embeddings(extracted_pretrained_json: str,
                                            corpus: Corpus,
                                            embedding_dim: int):
    emb_params = []

    print('Loading pretrained embeddings...')
    with open(extracted_pretrained_json) as file:
        word2vec_model = json.load(file)
    print('Loaded embeddings')

    found_in_word2vec = 0

    emb_params = np.random.uniform(-0.25, 0.25, (len(corpus.w2i), embedding_dim))

    for word, index in corpus.w2i.items():
        if word in word2vec_model:
            emb_params[index] = np.array(word2vec_model[word])
            found_in_word2vec += 1

    # 0th embedding should be set to 0, since it'll be used for padding.
    emb_params[0].fill(0)

    print('{} embeddings found in Word2Vec'.format(found_in_word2vec))
    return np.array(emb_params)

def word2vec_extract_pretrained_embeddings(word2vec_path:str,
                                           corpus: Corpus,
                                           output_json_path: str,
                                           binary=True):
    found_in_word2vec = 0
    word2vec_words = {}

    print('Loading pretrained embeddings...')
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)
    print('Loaded embeddings')

    for word in corpus.w2i.keys():
        if word in word2vec_model.vocab:
            word2vec_words[word] = word2vec_model[word].tolist()
            found_in_word2vec += 1

    print('{} embeddings found in Word2Vec'.format(found_in_word2vec))
    with open(output_json_path, 'w') as file:
        file.write(json.dumps(word2vec_words))
