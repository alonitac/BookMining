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
        sentence = sent_pos[i]
        noun_phrase = cp.parse(sent_pos[i])

        branch = []
        for subtree in noun_phrase.subtrees(filter=lambda t: t.label() == 'NP'):
            k = subtree.leaves()
            phrase = [k[i][0] for i in range(0, len(k))]
            separator = ', '
            phrase = separator.join(phrase)
            phrase = re.sub(', ', ' ', phrase)
            branch.append(phrase)

        # combine to or-criterion for re.sub replacement
        branch = separator.join(branch)
        criterion = re.sub(', ', ' | ', branch)
        criterion = ' ' + criterion + ' '

        # replace noun phrases by 'NP' token
        # TODO: criterion sometimes contains 'i', then also letter i is replaced by NOUNPHRASE. for example: which -> whNOUNPHRASEch, is -> NOUNPHRASEs
        #       solved it by adding spaces, that 'i' only gets replaced when it stands on it's own

        new_sentence = re.sub(criterion, ' NP ',
                              sentences[i])  # would not work if there is something like back NN,  back VBZ
        np_text.append(new_sentence)

    # first combine all sentences back to text and remove punctuation
    phrase = separator.join(np_text)
    text = re.sub(', ', ' ', phrase)
    space = '                                '  # 32 spaces
    clean_text = text.translate(str.maketrans(string.punctuation, space))

    return clean_text


def bar_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np):

    toks = [tokens, tokens_filtered, tokens_stemmed, tokens_np]

    fig1, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(16, 12))
    for ax, k in zip([ax1, ax2, ax3, ax4], toks):
        tokens_list = k[:50]
        people = tokens_list.index
        y_pos = np.linspace(len(people), 0)
        ax.barh(y_pos, tokens_list)
        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(people)
        ax.set_xlabel("Frequency of Token")

    ax1.set_title("unfiltered")
    ax2.set_title("filtered")
    ax3.set_title("filtered & stemmed")
    ax4.set_title("filtered, stemmed, noun phrase")

    plt.subplots_adjust(wspace=0.3)

    return fig1


def zipfian_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np):

    toks = [tokens, tokens_filtered, tokens_stemmed, tokens_np]

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip([ax1, ax2, ax3, ax4], toks):
        tokens_list = k
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
    ax4.set_title("filtered, stemmed, noun phrase")

    ax3.set_xlabel("Frequency rank of token")
    ax4.set_xlabel("Frequency rank of token")
    ax1.set_ylabel("Absolute frequency of token")
    ax3.set_ylabel("Absolute frequency of token")

    return fig1


if __name__ == '__main__':

    with open("Nietzsche.txt", "r") as f:
        plain_text = f.read().replace('\n', ' ')

    plain_words_lst = re.sub("[^\w]", " ", plain_text).split()

    space = '                                ' # 32 spaces
    clean_text = plain_text.translate(str.maketrans(string.punctuation, space)).lower()

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

    filtered_text = [w for w in tokenized_text if not w in stop_words]
    tokens_filtered = pd.Series(filtered_text).value_counts()

    # TODO: make more pretty
    print('tokens: ' + str(tokens[:20].index) +
          '\n filtered tokens: ' + str(tokens_filtered[:20].index))

    # (d) stemming

    ps = PorterStemmer()

    stemmed_text = [ps.stem(w) for w in filtered_text]
    tokens_stemmed = pd.Series(stemmed_text).value_counts()

    # nice to see that now philosoph = {philosophy, philosophers} and also moral = {moral, morality}

    # (e) noun phrase

    pt = plain_text[545:]

    clean_text = noun_phrases(pt)

    tokenized_text = nltk.word_tokenize(clean_text)
    tokens_np = pd.Series(tokenized_text).value_counts()

    # TODO: which version do we want to show?
    #       also change title of plot

    filtered_text = [w for w in tokenized_text if not w in stop_words]
    tokens_np = pd.Series(filtered_text).value_counts()

    stemmed_text = [ps.stem(w) for w in filtered_text]
    tokens_np = pd.Series(stemmed_text).value_counts()


    fig1 = bar_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np)
    fig2 = zipfian_plot(tokens, tokens_filtered, tokens_stemmed, tokens_np)

    plt.show()






# extract_noun_phrase does not work to 100 percent yet

extract_noun_phrase(sent_pos[3])

noun_phrases = [extract_noun_phrase(sent) for sent in sent_pos]

sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
                 ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
extract_noun_phrase(sentence) # missing: not only JJ, must be followed by noun

