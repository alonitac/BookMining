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
from collections import Counter
import matplotlib.pyplot as plt
import string
import numpy as np


def extract_noun_phrase(pos_list):
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    phrases = []
    i = 0
    while i < len(pos_list):
        if pos_list[i][1].startswith('JJ') or pos_list[i][1].startswith('NN'): # add: after JJ has to be JJ and at some point NN or NN directly
            phrase = pos_list[i][0]
            i += 1
            while i < len(pos_list) and pos_list[i][1].startswith('NN'):
                phrase += ' ' + pos_list[i][0]
                i += 1
            phrases.append(phrase)
        else:
            i += 1
    return phrases


def noun_phrases(plain_text):
    pt = plain_text
    space = '--'  # had problems with pos tags otherwise
    pt = pt.translate(str.maketrans('[]', space))
    pt = pt.lower()  # try if tags different when lowercase first "SUPPOSING" YES! SUPPOSING - NN, supposing - vbg
    sentences = sent_tokenize(pt)

    # pos for sentences
    sent_pos = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sentences]

    # find noun phrases
    grammar = r"""
      NP: {<JJ.*>+?<NN.*>+}
          {<NN.*>+}
    """

    cp = nltk.RegexpParser(grammar)

    np_text = []
    for i in range(0, len(sent_pos)):
        noun_phrase = cp.parse(sent_pos[i])

        branch = []
        for subtree in noun_phrase.subtrees(filter=lambda t: t.label() == 'NP'):
            k = subtree.leaves()
            phrase = [k[i][0] for i in range(0, len(k))]

            separator = ', '
            phrase = separator.join(phrase)
            phrase = re.sub(', ', ' ', phrase)
            branch.append(phrase)
        np_text = np_text + branch

    return ' '.join(np_text)


def bar_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np):

    toks = [tokens, tokens_filtered, tokens_stemmed, tokens_np]

    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(16, 12))
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

    ax1.set_title("unfiltered")
    ax2.set_title("filtered")
    ax3.set_title("filtered & stemmed")
    ax4.set_title("noun phrases")

    plt.subplots_adjust(wspace=0.3)

    return fig1


def zipfian_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np):

    toks = [tokens, tokens_filtered, tokens_stemmed, tokens_np]

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig1.subplots_adjust(hspace=0.3, wspace=0.1)

    for ax, k in zip([ax1, ax2, ax3, ax4], toks):
        tokens_list = k                             # TODO: rename
        counts = tokens_list
        indices = tokens_list.index
        ranks = np.arange(1, len(counts) + 1)
        frequencies = counts


        ax.loglog(ranks, frequencies, marker=".")

        for n in list(np.logspace(-0.5, np.log10(len(counts)), 20, endpoint=False).astype(int)):
            ax.text(ranks[n], frequencies[n], " " + indices[n],
                         verticalalignment="bottom",
                         horizontalalignment="left")

    ax1.set_title("unfiltered")
    ax2.set_title("filtered")
    ax3.set_title("filtered & stemmed")
    ax4.set_title("noun phrases")

    ax3.set_xlabel("Frequency Rank of Token")
    ax4.set_xlabel("Frequency Rank of Token")
    ax1.set_ylabel("Absolute Frequency of Token")
    ax3.set_ylabel("Absolute Frequency of Token")

    return fig1


if __name__ == '__main__':

    with open("Nietzsche.txt", "r") as f:
        plain_text = f.read().replace('\n', ' ')

    # TODO: Alon lowercase

    plain_words_lst = re.sub("[^\w]", " ", plain_text).split()

    space = '                                '  # 32 spaces
    clean_text = plain_text.translate(str.maketrans(string.punctuation, space)).lower()
    clean_words_lst = re.sub("[^\w]", " ", clean_text).split()

    # (b) tokenize text
    tokenized_text = nltk.word_tokenize(clean_text)
    tokens = pd.Series(tokenized_text).value_counts()



    # text_pos = nltk.pos_tag(plain_words_lst)
    # noun_phrases = extract_noun_phrase(text_pos)
    #
    # # start of (b) repeating
    # phrases_counts = Counter(noun_phrases[:50])  # I limit for first 50, otherwise it's not readable
    # df = pd.DataFrame.from_dict(phrases_counts, orient='index')
    # df.plot(kind='bar')
    # plt.show()


    # (c) stopwords

    # lowercase to also remove title and starting words
    stop_words = set(stopwords.words('english'))

    filtered_text = [w for w in clean_words_lst if w not in stop_words]
    filtered_text = ' '.join(filtered_text)
    tokens_filtered = nltk.word_tokenize(filtered_text)
    tokens_filtered = pd.Series(tokens_filtered).value_counts()

    # TODO: make more pretty
    print('tokens: ' + str(tokens[:20].index) +
          '\n filtered tokens: ' + str(tokens_filtered[:20].index))

    # (d) stemming

    ps = PorterStemmer()

    filtered_text = [w for w in clean_words_lst if w not in stop_words]
    stemmed_text = [ps.stem(w) for w in filtered_text]
    stemmed_text = ' '.join(stemmed_text)
    tokens_stemmed = nltk.word_tokenize(stemmed_text)
    tokens_stemmed = pd.Series(tokens_stemmed).value_counts()

    # nice to see that now philosoph = {philosophy, philosophers} and also moral = {moral, morality}

    # (e) noun phrase

    pt = plain_text[545:]
    pt = plain_text

    clean_text = noun_phrases(pt)

    tokenized_text = nltk.word_tokenize(clean_text)
    tokens_np = pd.Series(tokenized_text).value_counts()

    stemmed_text = [ps.stem(w) for w in tokenized_text]
    tokens_np = pd.Series(stemmed_text).value_counts()

    fig1 = bar_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np)
    fig2 = zipfian_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np)

    plt.show()

    # (f) faulty POS tagging
    sentence = 'SUPPOSING that Truth is a woman--what then?'
    sentence = 'WHOSE DUTY IS WAKEFULNESS ITSELF, are the heirs of all the strength which the struggle against this error has fostered.'
    nltk.pos_tag(nltk.word_tokenize(sentence))
    # tags 'SUPPOSING' as NN, but should be VBZ
    # TODO: Alans lowercase, then POS tag again and have right example hopefully.







#
#
# # extract_noun_phrase does not work to 100 percent yet
#
# extract_noun_phrase(sent_pos[3])
#
# noun_phrases = [extract_noun_phrase(sent) for sent in sent_pos]
#
# sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
#                  ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
# extract_noun_phrase(sentence) # missing: not only JJ, must be followed by noun
#
