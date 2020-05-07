import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from word2vec.w2v import Word2Vec
from RandomIndexing.random_indexing import RandomIndexing
import os

def draw_interactive(x, y, text):
    """
    Draw a plot visualizing word vectors with the posibility to hover over a datapoint and see
    a word associating with it
    
    :param      x:     A list of values for the x-axis
    :type       x:     list
    :param      y:     A list of values for the y-axis
    :type       y:     list
    :param      text:  A list of textual values associated with each (x, y) datapoint
    :type       text:  list
    """
    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn

    fig,ax = plt.subplots()
    sc = plt.scatter(x, y, c='b', s=100, cmap=cmap, norm=norm)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        note = "{}".format(" ".join([text[n] for n in ind["ind"]]))
        annot.set_text(note)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

class VecPlotter(object):
    def __init__(self, vec_file, vector_type, decomposition):
        self.__vec_source = vec_file
        self.__vector_type = vector_type
        self.__decomposition = decomposition
        print(self.__vec_source)
        print(self.__vector_type)
        print(self.__decomposition)

    def load_and_plot_vectors(self):
        if self.__vector_type == 'w2v':
            w2v = Word2Vec.load(self.__vec_source)
            word_vectors = w2v.get_word_vectors()
            text = w2v.get_words_for_wvs()

        if self.__decomposition == 'svd':
            dim_reducer = TruncatedSVD(n_components=2)

        if self.__decomposition == 'pca':
            dim_reducer = PCA(n_components=2)

        reduced_vectors = dim_reducer.fit_transform(word_vectors)

        x = reduced_vectors[:, 0]
        y = reduced_vectors[:, 1]
        draw_interactive(x, y, text)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding visualization toolkit')
    parser.add_argument('file', type=str, help='A textual file containing word vectors')
    parser.add_argument('-v', '--vector_type', default='w2v', choices=['w2v', 'ri'])
    parser.add_argument('-d', '--decomposition', default='svd', choices=['svd', 'pca'],
                        help='Your favorite decomposition method')
    args = parser.parse_args()

    # YOUR CODE HERE
    if os.path.exists(args.file):
        vec_plotter = VecPlotter(vec_file = args.file, vector_type = args.vector_type,
                                 decomposition = args.decomposition)
        vec_plotter.load_and_plot_vectors()



