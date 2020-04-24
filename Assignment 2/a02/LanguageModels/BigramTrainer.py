#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs
import json
import requests
from decimal import *

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = str(text_file.read()).lower()
        try :
            # Important that it is named self.tokens for the --check flag to work
            self.tokens = nltk.word_tokenize(text)
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE
        # update total_words
        self.total_words += 1

        # update unique_words and word-identifier mappings
        if token not in self.index:
            new_identifier = self.unique_words

            self.index[token] = new_identifier
            self.word[new_identifier] = token
            self.unique_words += 1


        # update unigram_count
        self.unigram_count[token] += 1

        # update bigram_count if we have processed at least two words
        if self.total_words > 1:
            last_word = self.word[self.last_index]
            self.bigram_count[last_word][token] += 1


        self.last_index = self.index[token]


    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        rows_to_print = []

        # YOUR CODE HERE
        rows_to_print.append(str(self.unique_words) + ' ' + str(self.total_words))

        for word in self.unigram_count:
            rows_to_print.append(str(self.index[word]) + ' ' + word + ' ' + str(self.unigram_count[word]))

        # print bigram log-probabilities
        for bigram_1 in self.bigram_count:
            for bigram_2 in self.bigram_count[bigram_1]:
                bigram_count = self.bigram_count[bigram_1][bigram_2]
                bigram_prob = bigram_count / self.unigram_count[bigram_1]
                bigram_logprob = "%.15f" % round(math.log(bigram_prob), 15)
                bigram_1_id = self.index[bigram_1]
                bigram_2_id = self.index[bigram_2]
                rows_to_print.append(str(bigram_1_id) + ' ' + str(bigram_2_id) + ' ' + str(bigram_logprob))

        rows_to_print.append('-1')

        return rows_to_print

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')
    parser.add_argument('--check', action='store_true', help='check if your alignment is correct')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    if arguments.check:
        results  = bigram_trainer.stats()
        payload = json.dumps({
            'tokens': bigram_trainer.tokens,
            'result': results
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab2_trainer',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print('Success! Your results are correct')
            for row in results: print(row)
        else:
            print('Your results:\n')
            for row in results: print(row)
            print("The server's results:\n")
            for row in response_data['result']: print(row)
    else:
        stats = bigram_trainer.stats()
        if arguments.destination:
            with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
                for row in stats: f.write(row + '\n')
        else:
            for row in stats: print(row)


if __name__ == "__main__":
    main()
