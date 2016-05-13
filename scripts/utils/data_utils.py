
import collections
import csv
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pickle
import sys

DATA_DIR = '../game_summarization'
TITLE_TOKENS_FILENAME = 'titles.p'
TITLE_NUMERIC_LABLES_FILENAME = 'title_numeric.csv'
TITLE_WORD_LABLES_FILENAME = 'title_words.csv'

def plot_word_count_histogram():
    # plt.hist(vocab.values(), 1000)
    # plt.xlim(0, 200)
    # plt.show()
    pass

def preprocess(entities):
    """ 
    Given a list of lists of words, return those lists with preprocessing applied.
    For example, stemming, removing words with too low a count, combining numbers,
    etc.

    Returns a list of lists of words
    """
    # get counts
    # raw_vocab = collections.defaultdict(lambda: 0)
    # for entity in entities:
    #     for word in entity:
    #         raw_vocab[word] += 1
    return entities
    
def get_word_to_label_mappings(entities):
    """
    Given a list of lists of words, return a dictionary mapping a word to
    and index / label for that word.
    """
    words = set()
    for entity in entities:
        for word in entity:
            words.add(word)
    alpha_sorted_words = sorted(words)

    labels_to_words = dict()
    words_to_labels = dict()
    for idx, word in enumerate(alpha_sorted_words):
        words_to_labels[word] = idx
        labels_to_words[idx] = word
    return words_to_labels, labels_to_words

def convert_to_labels(entities, word_to_index):
    label_entities = []
    for entity in entities:
        label_entity = []
        for word in entity:
            label = word_to_index[word]
            label_entity.append(label)
        label_entities.append(label_entity)
    return label_entities

def convert_tokens_to_numeric_labels(input_filepath, output_filepath):
    entities = np.array(pickle.load(open(input_filepath, "rb")))
    preprocessed_entities = preprocess(entities)
    words_to_labels, labels_to_words = get_word_to_label_mappings(preprocessed_entities)
    label_entities = convert_to_labels(preprocessed_entities, words_to_labels)
    with open(output_filepath, 'wb') as outfile:
        for labels in label_entities:
            for widx in labels:
                outfile.write(labels_to_words[widx].lower().encode('utf8')),
                outfile.write(' ')
            outfile.write('\n')

if __name__ == '__main__':
    title_tokens_filepath = os.path.join(DATA_DIR, TITLE_TOKENS_FILENAME)
    title_numeric_labels_output_filepath = os.path.join(DATA_DIR, TITLE_WORD_LABLES_FILENAME)
    convert_tokens_to_numeric_labels(title_tokens_filepath, title_numeric_labels_output_filepath)