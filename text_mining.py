import nltk
# uncomment the following lines if nltk asks for these packages
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
import re
import pandas as pd
import matplotlib.pyplot as plt
import string
import numpy as np


# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

def noun_phrases(plain_text):
    """
    Extracts all the noun phrases from a text.
    :param plain_text: the whole text in one string
    :return: noun_phrases: all the noun phrases in one string without separation
    """

    # replace brackets by dashes, had problems with pos tags otherwise
    space = '--'
    plain_text = plain_text.translate(str.maketrans('[]', space))
    sentences = sent_tokenize(plain_text)

    # list of pos tagged sentences
    sent_pos = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sentences]

    # find noun phrases
    # create grammar rule to search for noun phrases
    grammar = r"""
      NP: {<JJ.*>+?<NN.*>+}
          {<NN.*>+}
    """
    cp = nltk.RegexpParser(grammar)

    np_text = []
    for i in range(0, len(sent_pos)):
        # apply grammar rule to find the noun phrases, returns Trees
        noun_phrase = cp.parse(sent_pos[i])

        branch = []
        # extract noun phrases from Tree
        for subtree in noun_phrase.subtrees(filter=lambda t: t.label() == 'NP'):
            k = subtree.leaves()
            phrase = [k[i][0] for i in range(0, len(k))]
            separator = ', '
            phrase = separator.join(phrase)
            phrase = re.sub(', ', ' ', phrase)
            branch.append(phrase)
        np_text = np_text + branch

    noun_phrases = ' '.join(np_text)  # bring in desired format

    return noun_phrases


def bar_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np):
    """
    Creates bar plots for the 40 most frequent tokens
    :param tokens: tokens of whole text, with stopwords
    :param tokens_filtered: tokens of text without stopwords
    :param tokens_stemmed: tokens of text without stopwords and stemmed
    :param tokens_np: tokens of noun phrases only
    :return: bar plots
    """

    toks = [tokens, tokens_filtered, tokens_stemmed, tokens_np]

    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(12, 12))
    for ax, k in zip([ax1, ax2, ax3, ax4], toks):
        tokens_list = k[:40]

        people = tokens_list.index
        y_pos = np.arange(1, len(people) + 1)
        y_pos = np.abs(np.sort(-y_pos))

        ax.barh(y_pos, tokens_list)
        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(people)
        ax.set_xlabel("Frequency of Token")

    ax1.set_title("(b) whole text")
    ax2.set_title("(c) w/o stopwords")
    ax3.set_title("(d) w/o stopwords & stemmed")
    ax4.set_title("(e) noun phrases, stemmed")

    plt.subplots_adjust(wspace=0.3)
    fig1.tight_layout()

    return fig1


def zipfian_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np):
    """
    Creates log-Frequency,log-Rank plot to analyze word distribution according to zipfians law
    :param tokens: tokens of whole text, with stopwords
    :param tokens_filtered: tokens of text without stopwords
    :param tokens_stemmed: tokens of text without stopwords and stemmed
    :param tokens_np: tokens of noun phrases only
    :return: log-log plots
    """

    toks = [tokens, tokens_filtered, tokens_stemmed, tokens_np]

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig1.subplots_adjust(hspace=0.3, wspace=0.1)

    for ax, tokens_list in zip([ax1, ax2, ax3, ax4], toks):
        indices = tokens_list.index   # string of token, need for text
        ranks = np.arange(1, len(tokens_list) + 1)
        frequencies = tokens_list

        ax.loglog(ranks, frequencies, marker=".")

        # print some words to their points
        for n in list(np.logspace(-0.5, np.log10(len(tokens_list)), 17, endpoint=False).astype(int)):
            ax.text(ranks[n], frequencies[n], " " + indices[n],
                         verticalalignment="bottom",
                         horizontalalignment="left")

    ax1.set_title("(b) whole text")
    ax2.set_title("(c) w/o stopwords")
    ax3.set_title("(d) w/o stopwords & stemmed")
    ax4.set_title("(e) noun phrases, stemmed")

    ax3.set_xlabel("Rank of Token")
    ax4.set_xlabel("Rank of Token")
    ax1.set_ylabel("Absolute Frequency of Token")
    ax3.set_ylabel("Absolute Frequency of Token")
    fig1.tight_layout()

    return fig1


def to_lowercase_words(text):
    """
    Cleans text. If a word with more than one letters is written in all uppercase letters,
    the function lowercases all letters but the first one
    :param text: string of text
    :return: cleaned string of text
    """
    return re.sub(r"[A-Z][A-Z]+", lambda word: word.group(0).lower(), text)


if __name__ == '__main__':

    # load book and clean the text
    with open("Nietzsche.txt", "r") as f:
        plain_text = f.read().replace('\n', ' ')

    plain_text = to_lowercase_words(plain_text) # lowercase the words that are written in all capitals
    plain_words_lst = re.sub("[^\w]", " ", plain_text).split()

    # remove punctuation for task (b), (c), (d)
    space = '                                '  # 32 spaces
    clean_text = plain_text.translate(str.maketrans(string.punctuation, space))
    clean_words_lst = re.sub("[^\w]", " ", clean_text).split()

    # (b) tokenize text
    tokenized_text = nltk.word_tokenize(clean_text)
    tokens = pd.Series(tokenized_text).value_counts()  # count occurrence

    # (c) stopwords

    stop_words = set(stopwords.words('english'))

    # filter the text and only keep words that are not stopwords.
    # all words in stop_words are lower case so also lowercase word when lookup
    filtered_text = [w for w in clean_words_lst if w.lower() not in stop_words]
    filtered_text = ' '.join(filtered_text)
    tokens_filtered = nltk.word_tokenize(filtered_text)
    tokens_filtered = pd.Series(tokens_filtered).value_counts()  # count occurrence

    # (d) stemming

    ps = PorterStemmer()

    # also here remove stopwords
    filtered_text = [w for w in clean_words_lst if w.lower() not in stop_words]
    # and stemm the words
    stemmed_text = [ps.stem(w) for w in filtered_text]
    stemmed_text = ' '.join(stemmed_text)
    tokens_stemmed = nltk.word_tokenize(stemmed_text)
    tokens_stemmed = pd.Series(tokens_stemmed).value_counts()  # count occurrence

    # (e) noun phrase

    # extract noun phrases from text, stemm and tokenize them
    np_text = noun_phrases(plain_text)

    tokenized_text = nltk.word_tokenize(np_text)
    stemmed_text = [ps.stem(w) for w in tokenized_text]
    tokens_np = pd.Series(stemmed_text).value_counts()


    print('20 most frequent tokens: \n' +
          'whole text: \n' + str(tokens[:20].index.values) +
          '\n w/o stopwords: \n' + str(tokens_filtered[:20].index.values) +
          '\n w/o stopwords & stemmed: \n' + str(tokens_stemmed[:20].index.values) +
          '\n noun phrases, stemmed: \n' + str(tokens_np[:20].index.values))

    # create plots
    fig1 = bar_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np)
    fig2 = zipfian_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np)
    fig1.savefig('tokens.png')
    fig2.savefig('zipfian_graph.png')

    plt.show()

    # (f) faulty POS tagging
    sentence = 'the reverse of those hitherto prevalent--philosophers of the dangerous "Perhaps" in every sense of the term'
    sentence = 'hitherto prevalent'
    sentence = 'WHOSE DUTY IS WAKEFULNESS ITSELF, are the heirs of all the strength which the struggle against this error has fostered.'
    sentence = 'WHAT really is this "Will to Truth" in us?'
    sentence = 'The fundamental belief of metaphysicians is THE BELIEF IN ANTITHESES OF VALUES.'

    sentence_lc = to_lowercase_words(sentence)
    # Actual: [('hitherto', 'NN'), ('prevalent', 'NN')]
    # Expected: [('hitherto', 'RB'), ('prevalent', 'JJ')]
    print(nltk.pos_tag(nltk.word_tokenize(sentence)))
    print(nltk.pos_tag(nltk.word_tokenize(sentence_lc)))