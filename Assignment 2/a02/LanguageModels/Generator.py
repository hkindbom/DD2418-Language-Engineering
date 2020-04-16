import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # parsing language model
                lines = f.readlines()
                for line in lines[:self.unique_words]:
                    line = line.strip().split(' ')
                    self.word[int(line[0])] = line[1]
                    self.index[line[1]] = int(line[0])
                    self.unigram_count[line[1]] = int(line[2])

                for line in lines[self.unique_words:len(lines) - 1]:
                    line = line.strip().split(' ')
                    bigram_1 = self.word[int(line[0])]
                    bigram_2 = self.word[int(line[1])]
                    self.bigram_prob[bigram_1][bigram_2] = float(line[2])

                # REUSE YOUR CODE FROM BigramTester.py here
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and following the distribution
        of the language model.
        """ 
        # YOUR CODE HERE
        w = w.lower()
        print(w)

        for word in range(n):
            w = self.get_next_word(w)
            print(w)


    def get_next_word(self,w):
        bigrams = self.bigram_prob[w]
        pos_next_word = list(bigrams.keys())
        next_word_logprob = list(bigrams.values())
        next_word_probs = list(map(lambda x: math.exp(x), next_word_logprob))

        # generate next word with uniform dist if current word doesnt exist in vocab
        # or bigram has very low prob
        if not bigrams or max(next_word_probs) < 0.000000000001:
            return self.get_random_word()

        next_word_choice = random.choices(pos_next_word, next_word_probs)[0]
        return next_word_choice

    def get_random_word(self):
        random_word = random.choice(list(self.unigram_count.keys()))
        return random_word


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
