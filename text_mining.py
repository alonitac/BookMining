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
        if pos_list[i][1].startswith('JJ') or pos_list[i][1].startswith('NN'):
            phrase = pos_list[i][0]
            i += 1
            while i < len(pos_list) and pos_list[i][1].startswith('NN'):
                phrase += ' ' + pos_list[i][0]
                i += 1
            phrases.append(phrase)
        else:
            i += 1
    return phrases


if __name__ == '__main__':

    with open("Nietzsche.txt", "r") as f:
        plain_text = f.read().replace('\n', ' ')
        plain_words_lst = re.sub("[^\w]", " ", plain_text).split()

        space = '                                ' # 32 spaces
        clean_text = plain_text.translate(str.maketrans(string.punctuation, space)).lower()

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

        # start of (b)

        # lowercase_text = [w.lower() for w in tokenized_text]
        # tokens_lc = pd.Series(lowercase_text).value_counts()

        # (c) stopwords

        # lowercase to also remove title and starting words  # TODO: already do this in the beginning?
                                                             #  right now the tokens are lowercased,
                                                             #  not the text itself. prob better to do it right in the beginning
        # TODO: remove punctuation 25061
        # TODO: frequency in tokens not count freq=count/total

        # lowercase_text = [w.lower() for w in tokenized_text]
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


        fig1, (ax0, ax1, ax2) = plt.subplots(ncols=3, nrows=1, figsize=(16, 12))

        ax0.set_title('unfiltered')
        people = tokens[:50].index
        y_pos = np.linspace(len(people),0)
        ax0.barh(y_pos, tokens[:50])
        ax0.set_xscale('log')
        ax0.set_xlabel('occurrence')
        ax0.set_yticks(y_pos)
        ax0.set_yticklabels(people)

        ax1.set_title('filtered')
        people = tokens_filtered[:50].index
        y_pos = np.linspace(len(people),0)
        ax1.barh(y_pos, tokens_filtered[:50])
        ax1.set_xscale('log')
        ax1.set_xlabel('occurrence')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(people)

        ax2.set_title('filtered & stemmed')
        people = tokens_stemmed[:50].index
        y_pos = np.linspace(len(people),0)
        ax2.barh(y_pos, tokens_stemmed[:50])
        ax2.set_xscale('log')
        ax2.set_xlabel('occurrence')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(people)

        plt.subplots_adjust(wspace=0.3)

        plt.show()




        # (d) sentences



    # A Zipf plot
    counts = tokens_stemmed
    indices = tokens_stemmed.index

    ranks = np.arange(1, len(counts)+1)
    #indices = np.argsort(-counts)
    #frequencies = counts[indices]

    frequencies = counts

    plt.loglog(ranks, frequencies, marker=".")
    plt.title("Zipf plot for Nietzsche tokens")
    plt.xlabel("Frequency rank of token")
    plt.ylabel("Absolute frequency of token")
    plt.grid(True)

    for n in list(np.logspace(-0.5, np.log10(len(counts)), 20).astype(int)):
        print(n)

        dummy = plt.text(ranks[n], frequencies[n], " " + indices[n],
                     verticalalignment="bottom",
                     horizontalalignment="left")


# (d) sentences

pt = plain_text[545:]
sentences = sent_tokenize(pt)

period_sent = [re.split('\. ', sent) for sent in sentences]
check_period = [group for group in period_sent if len(group)>1]

# does good job
check_period[1] # '...' but sometimes it should also end like this...
check_period[3] # 'St. Augustine'
check_period[5] # '92. Who has not' list
check_period[6] # 'Nov' abriviation
check_period[8] # 'Arther B. Bullock' names

# pos for sentences
sent_pos = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sentences]

extract_noun_phrase(sent_pos[3])


len(period_sent)
len(sent_tokenize_list)

