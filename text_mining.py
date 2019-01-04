import nltk
# uncomment the following lines if nltk asks for these packages
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


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

        text_pos = nltk.pos_tag(plain_words_lst)
        noun_phrases = extract_noun_phrase(text_pos)

        # start of (b) repeating
        phrases_counts = Counter(noun_phrases[:50])  # I limit for first 50, otherwise it's not readable
        df = pd.DataFrame.from_dict(phrases_counts, orient='index')
        df.plot(kind='bar')
        plt.show()

        # start of (b)
        tokenized_text = nltk.word_tokenize(plain_text)

