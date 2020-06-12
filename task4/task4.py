"""
 Сделать 2 модели векторизации на базе word-context или word-document подхода, векторизовать ими слова из wordsim353,
 поиграться с их параметрами для улучшения метрики корреляции Пирсона с оценками экспертов и для самой лучшей модели
  посмотреть как на качество влияет выбор меры сравнения векторов
  ver2:
  Alexey Platonov, [15.04.20 18:50]
Четвертое задание - запрогать один из классических алгоритмов векторизации на матрицах (bag of words, hal, term-document
 matrix, pmi word-word matrix) и настроить их параметры по wordsim353

Alexey Platonov, [15.04.20 18:51]
Поправка не один, а два алгоритма - и сравнить их между собой после тюнинга на wordsim
"""

import math
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from scipy import linalg
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.spatial.distance import dice
import itertools

lemmatizer = WordNetLemmatizer()
stemming = PorterStemmer()


def get_sentences(folder):
    # ref: 3th lab
    res = []
    file_container = [raw_file for raw_file in os.walk(folder)]
    file_container = file_container[0][2]
    for files in file_container:
        with open(f"{folder}/{files}", "r", encoding="UTF-8") as f:
            sent_list = nltk.sent_tokenize(f.read())
            res += sent_list
    return res


sents = get_sentences("./texts")


def co_occurrence(sentences, window_size=2):
    # ref: https://stackoverflow.com/questions/58701337/how-to-construct-ppmi-matrix-from-a-text-corpus
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        words = nltk.word_tokenize(text)
        words = list(set([re.sub(r'[-/{}\[\].“―_,:?!…`"—\'0-9()⚡]', "", w.lower()) for w in words if len(w) > 1 and w
                          not in stopwords.words('english')]))
        for i in range(len(words)):
            token = words[i]
            vocab.add(token)
            next_token = words[i + 1: i + 1 + window_size]
            for t in next_token:
                key = tuple(sorted([t, token]))
                d[key] += 1

    vocab = sorted(vocab)
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df


df = co_occurrence(sents, 1)


def pmi(dataframe):
    """ very slow """
    col_totals = dataframe.sum(axis=0)
    total = col_totals.sum()
    row_totals = dataframe.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    dataframe = dataframe / expected
    with np.errstate(divide='ignore'):
        dataframe = np.log(dataframe)
    dataframe[np.isinf(dataframe)] = 0.0
    dataframe[dataframe < 0] = 0.0
    return dataframe


ppmi = pmi(df)
keys_pmi = ppmi.keys()


def lsa(input_matrix):
    """ very slow """
    # ref https://web.stanford.edu/~jurafsky/li15/lec3.vector.pdf
    U, s, Vh = linalg.svd(input_matrix)
    trans_matrix = np.dot(np.dot(U, linalg.diagsvd(s, len(input_matrix), len(Vh))), Vh)
    return pd.DataFrame(data=trans_matrix, columns=input_matrix.keys(), index=input_matrix.keys())


lsa = lsa(df)
keys_lsa = lsa.keys()


def import_wordsim_words(path):
    """ if finds any word in corpus it  """
    output_list_1 = []
    golden_witch = []
    with open(path, "r") as wordsim_file:
        raw_wordsim_data = wordsim_file.readlines()
        for words in raw_wordsim_data:
            words_processed = words.split("\t")
            if words_processed[0] in keys_pmi and words_processed[1] in keys_pmi:
                print(words_processed)
                output_list_1.append(words_processed[:2])
                golden_witch.append(words_processed[2])
        return output_list_1, golden_witch


wordlist, beato = import_wordsim_words("./wordsim353_sim_rel/wordsim_similarity_goldstandard.txt")
print(wordlist)


def process_ws(wlist):
    """ will take a decade to compute """
    with open("cosine.txt", "w+") as cosine_file, open("dice.txt", "w+") as dice_file:
        cosine_file.write("pair\t\t\t\t\tppmi lsa golden\n")
        dice_file.write("pair\t\t\t\t\tppmi lsa golden\n")
        for i, pair in enumerate(wlist):
            pmi_vector_1 = np.array([ppmi.get(pair[0]).values])
            pmi_vector_2 = np.array([ppmi.get(pair[1]).values])
            pmi_vector_1 = pmi_vector_1[~np.isnan(pmi_vector_1)]
            pmi_vector_2 = pmi_vector_2[~np.isnan(pmi_vector_2)]
            ppmi_number_cos = cosine_similarity(pmi_vector_1.reshape(1, -1), pmi_vector_2.reshape(1, -1))[0][0]
            ppmi_number_dice = dice(pmi_vector_1, pmi_vector_2)

            lsa_vector_1 = np.array([lsa.get(pair[0]).values])
            lsa_vector_2 = np.array([lsa.get(pair[1]).values])
            lsa_vector_1 = lsa_vector_1[~np.isnan(lsa_vector_1)]
            lsa_vector_2 = lsa_vector_2[~np.isnan(lsa_vector_2)]
            lsa_number_cos = cosine_similarity(lsa_vector_1.reshape(1, -1), lsa_vector_2.reshape(1, -1))[0][0]
            lsa_number_dice = dice(lsa_vector_1, lsa_vector_2)

            cosine_file.write(f"{pair}\t{round(ppmi_number_cos, 3)}\t{round(lsa_number_cos, 3)}\t{beato[i]}")
            dice_file.write(f"{pair}\t{round(ppmi_number_dice, 3)}\t{round(lsa_number_dice, 3)}\t{beato[i]}")


process_ws(wordlist)