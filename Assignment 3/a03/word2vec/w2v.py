import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected # False => use "usual" unigram distribution
        self.__use_lr_scheduling = use_lr_scheduling
        self.__unigram_prob = {}
        self.__corr_unigram_prob = {}
        self.__unigram_count = {}
        self.__V = 0 # Number of unique words in corpus
        self.__tot_words = 0
        self.__unigram_dist_words = []
        self.__unigram_dist_probs = []
        self.__corr_unigram_dist_words = []
        self.__corr_unigram_dist_probs = []
        self.__weight_init_uniform = True  # False => normal distribution
        self.__normal_mu = 0
        self.__normal_sigma = 2
        self.__processed_words = 0
        self.__nn_model = None
        self.__k = 5


    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w


    @property
    def vocab_size(self):
        return self.__V
        

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        #
        # REPLACE WITH YOUR CODE HERE
        #
        return re.sub(r'\d|[^\w\s]', '', line).lower().split()


    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        #
        # REPLACE WITH YOUR CODE
        #
        context = []

        nr_words_before = i
        nr_words_after = len(sent) - 1 - i

        nr_right_context_words = min(nr_words_after, self.__rws)
        nr_left_context_words = min(nr_words_before, self.__lws)

        # Append right context indices
        for right_word in sent[i + 1: i + nr_right_context_words + 1]:
            context.append(self.__w2i[right_word])

        # Append left context indices
        for left_pos in range(1, nr_left_context_words + 1):
            left_word = sent[i - left_pos]
            context.append(self.__w2i[left_word])

        return context


    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        #
        # REPLACE WITH YOUR CODE
        #
        self.__w2i = {}
        self.__i2w = []

        focus_words = []
        context_words = []

        # Loop through all lines of word lists in all textfiles
        line_gen = self.text_gen()
        for word_list in line_gen:
            self.build_word_idx_maps(word_list)
            for focus_idx, focus_word in enumerate(word_list):
                focus_words.append(focus_word)
                context_words.append(self.get_context(word_list, focus_idx))

        self.calc_unigram_dist()
        self.calc_corr_unigram_dist()

        return focus_words, context_words

    def build_word_idx_maps(self, word_list):
        for word in word_list:
            self.__tot_words += 1
            if word not in self.__unigram_count:
                self.__V += 1
                self.__unigram_count[word] = 1
                self.__w2i[word] = len(self.__i2w)
                self.__i2w.append(word)
            else:
                self.__unigram_count[word] += 1

    def calc_unigram_dist(self):
        for unique_word in self.__unigram_count:
            self.__unigram_prob[unique_word] = self.__unigram_count[unique_word] / self.__tot_words

    def calc_corr_unigram_dist(self):
        power_sum = self.calc_power_sum()
        for unique_word in self.__unigram_count:
            unigram_prob = self.__unigram_prob[unique_word]
            self.__corr_unigram_prob[unique_word] = unigram_prob ** (3 / 4) / power_sum

    def calc_power_sum(self):
        power_sum = 0
        for unique_word in self.__unigram_count:
            power_sum += self.__unigram_prob[unique_word] ** (3 / 4)
        return power_sum

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        #
        # REPLACE WITH YOUR CODE
        #
        found_replicate = True
        # Resample without replacement as long as focus/context word in negative samples
        while found_replicate:
            if self.__use_corrected:
                negative_samples = np.random.choice(self.__corr_unigram_dist_words, number, replace=False, p = self.__corr_unigram_dist_probs).tolist()
            else:
                negative_samples = np.random.choice(self.__unigram_dist_words, number, replace=False, p = self.__unigram_dist_probs).tolist()

            if self.__i2w[xb] not in negative_samples and self.__i2w[pos] not in negative_samples:
                found_replicate = False

        return negative_samples

    def build_dist_lists(self):
        self.__unigram_dist_words = list(self.__unigram_prob.keys())
        self.__unigram_dist_probs = list(self.__unigram_prob.values())
        self.__corr_unigram_dist_words = list(self.__corr_unigram_prob.keys())
        self.__corr_unigram_dist_probs = list(self.__corr_unigram_prob.values())

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        print("Dataset contains {} datapoints".format(N))

        # Convert dicts of unigram probs to lists enabling np.random.choice
        self.build_dist_lists()

        # RANDOM INITIALIZATION OF WORD VECTORS
        if self.__weight_init_uniform:
            self.__W = np.random.rand(self.__V, self.__H)
            self.__U = np.random.rand(self.__V, self.__H)

        else:
            self.__W = np.random.normal(self.__normal_mu, self.__normal_sigma, (self.__V, self.__H))
            self.__U = np.random.normal(self.__normal_mu, self.__normal_sigma, (self.__V, self.__H))

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):
                #
                # YOUR CODE HERE 
                #
                focus_word = x[i]
                focus_word_idx = self.__w2i[focus_word]
                pos_sample_indices = t[i]
                pos_samples = self.indices_to_words(pos_sample_indices)
                neg_samples = self.get_neg_samples(focus_word_idx, pos_sample_indices)

                self.gradient_descent(focus_word_idx, pos_samples, neg_samples)

                self.__processed_words += 1

            print('Epoch ', ep, ' finished')

    def gradient_descent(self, focus_word_idx, pos_samples, neg_samples):

        focus_word_gradient = self.gradient_wrt_focus_vec(focus_word_idx, pos_samples, neg_samples)
        neg_samples_gradients_dict = self.gradients_wrt_neg_vec(focus_word_idx, neg_samples)
        pos_samples_gradients_dict = self.gradients_wrt_pos_vec(focus_word_idx, pos_samples)

        self.__W[focus_word_idx] -= self.__lr * focus_word_gradient

        self.upd_sample_vecs(neg_samples_gradients_dict)
        self.upd_sample_vecs(pos_samples_gradients_dict)

        self.upd_learning_rate()

    def upd_learning_rate(self):
        if self.__lr < self.__init_lr * 0.0001:
            self.__lr = self.__init_lr * 0.0001
            return
        self.__lr = self.__init_lr * (1 - self.__processed_words / (self.__epochs * self.__tot_words + 1))

    def upd_sample_vecs(self, gradients):

        for word_idx, gradient in gradients.items():
            self.__U[word_idx] -= self.__lr * gradient

    def gradient_wrt_focus_vec(self, focus_word_idx, pos_samples, neg_samples):
        focus_gradient = np.zeros(self.__H)
        focus_vec = self.__W[focus_word_idx]

        for pos_sample in pos_samples:
            pos_vec = self.__U[self.__w2i[pos_sample]]
            focus_gradient += pos_vec * (self.sigmoid(np.dot(pos_vec, focus_vec)) - 1)

        for neg_sample in neg_samples:
            neg_vec = self.__U[self.__w2i[neg_sample]]
            focus_gradient += neg_vec * self.sigmoid(np.dot(neg_vec, focus_vec))

        return focus_gradient

    def gradients_wrt_neg_vec(self, focus_word_idx, neg_samples):
        neg_samples_gradients_dict = {}
        focus_vec = self.__W[focus_word_idx]
        for neg_sample in neg_samples:
            neg_vec = self.__U[self.__w2i[neg_sample]]
            neg_samples_gradients_dict[self.__w2i[neg_sample]] = focus_vec * self.sigmoid(np.dot(neg_vec, focus_vec))

        return neg_samples_gradients_dict

    def gradients_wrt_pos_vec(self, focus_word_idx, pos_samples):
        pos_samples_gradients_dict = {}
        focus_vec = self.__W[focus_word_idx]
        for pos_sample in pos_samples:
            pos_vec = self.__U[self.__w2i[pos_sample]]
            pos_samples_gradients_dict[self.__w2i[pos_sample]] = focus_vec * (self.sigmoid(np.dot(pos_vec, focus_vec)) - 1)

        return pos_samples_gradients_dict

    def get_neg_samples(self, focus_word_idx, pos_sample_indices):
        neg_samples = []
        for pos_sample_idx in pos_sample_indices:
            neg_samples.extend(self.negative_sampling(self.__nsample, focus_word_idx, pos_sample_idx))
        return neg_samples

    def indices_to_words(self, indices):
        words = []
        for idx in indices:
            words.append(self.__i2w[idx])
        return words

    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        #
        # REPLACE WITH YOUR CODE
        #
        # Create nearest neighbour model if not exists already
        if self.__nn_model is None:
            # look on parameters
            self.__nn_model = NearestNeighbors(self.__k, metric=metric)
            self.__nn_model.fit(self.__W)

        query_vecs = self.get_query_vecs(words)

        distances, word_indices = self.__nn_model.kneighbors(query_vecs)
        formated_nn = self.format_nearest_neighbours(distances, word_indices)

        return formated_nn

    def get_query_vecs(self, words):
        query_vecs = np.zeros([len(words), self.__H])
        for index, word in enumerate(words):
            query_vecs[index] = self.__W[self.__w2i[word.lower()]]

        return query_vecs

    def format_nearest_neighbours(self, distances, word_indices):
        formated_nn = []
        for distance_list, word_index_list in zip(distances, word_indices):
            nn_data = []
            for distance, word_index in zip(distance_list, word_index_list):
                nn_word = self.__i2w[word_index]
                nn_data.append((nn_word, round(distance, 3)))
            # Sort nearest neighbour list in descending similarity order
            nn_data.sort(key=lambda x: x[1])
            formated_nn.append(nn_data)

        return formated_nn


    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    f.write(str(w) + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
        except:
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v

    def get_word_vectors(self):
        return self.__W

    def get_words_for_wvs(self):
        words = []
        for word_index, word_vec in enumerate(self.__W):
            words.append(self.__i2w[word_index])
        return words

    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=int(args.epochs),
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
