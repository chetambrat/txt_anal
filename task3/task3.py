"""

 1) спроектировать модуль хранения и чтения словарей.
 2) на английском или на русском собрать 3 корпуса текста (тематическая выборка – 2 штуки (выбрать две тематики на одном
  языке и собрать 20-30 текстов (не твиты, но при этом не глава «Война и мир», что-то среднее, размерности статьи на
   хабре) по каждой тематике + «просто» тексты привести к одному формату).
 3) спроектировать словарь стоп слов (три выборки сливаем в один документ и рассчитываем TF-IDF для каждого слова,
  отфильтровать и посмотреть, что получилось).
 4) для первых двух выборок (тематических) контрастным методом найти специфические термины, характерные для данных
  выборок (выбрасываем стоп-слова и смотрим слова, которые имеют высокий вес для одной выборки и низкий вес для другой
   выборки). Можно составить словарь специфических терминов.
"""

import math
import os
import re
from collections import Counter

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stemming = PorterStemmer()
stop_words = []


def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count / sum_nk
    return tf


def compute_idf(strings_list):
    n = len(strings_list)
    idf = dict.fromkeys(strings_list[0].keys(), 0)
    for l in strings_list:
        for word, count in l.items():
            if count > 0:
                if word in idf:
                    idf[word] += 1
                if word not in idf:
                    idf[word] = 1

    for word, v in idf.items():
        idf[word] = math.log10(n / float(v))
    return idf


def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf


def import_raw_text(folder):
    result_list = []
    file_container = [raw_file for raw_file in os.walk(folder)]
    file_container = file_container[0][2]
    for files in file_container:
        with open(f"{folder}/{files}", "r", encoding="UTF-8") as f:
            word_list = nltk.word_tokenize(f.read())
            word_list = list(
                set([re.sub(r'[-/{}\[\].“,:?!`"—\'0-9()]', "", w.lower()) for w in word_list if len(w) > 1]))
            result_list.append(word_list)
    return result_list


def count_word_dict(list_of_words):
    word_dict_list = []
    for document in list_of_words:
        word_dict = {}
        for word_in_corpus in document:
            if word_in_corpus in word_dict:
                word_count = word_dict.get(word_in_corpus)
                word_dict[word_in_corpus] = word_count + 1
            if word_in_corpus not in word_dict:
                word_dict[word_in_corpus] = 1
        word_dict_list.append(word_dict)
    return word_dict_list


def unpack_lists(packed_list):
    flat_list = [item for sublist in packed_list for item in sublist]
    return flat_list


ccp_list = import_raw_text("./CCP")
gg_list = import_raw_text("./GamerGate")
rnd_list = import_raw_text("./random")

ccp_dict = count_word_dict(ccp_list)
gg_dict = count_word_dict(gg_list)
rnd_dict = count_word_dict(rnd_list)


def zip_tf(dict_obj, list_obj):
    tf_list = []
    for i in zip(dict_obj, list_obj):
        tf_dict = compute_tf(i[0], i[1])
        tf_list.append(tf_dict)
    return tf_list


def zip_tfidf(tf_obj, idf_obj):
    tfidf_list = []
    for i in tf_obj:
        tfidf_dict = compute_tf_idf(i, idf_obj)
        tfidf_list.append(tfidf_dict)
    return tfidf_list


ccp_tf = zip_tf(ccp_dict, ccp_list)
gg_tf = zip_tf(gg_dict, gg_list)
rnd_tf = zip_tf(rnd_dict, rnd_list)

merged_dict_1, merged_dict_2, merged_dict_3 = {}, {}, {}
total_dict = {}
for _ in ccp_dict:
    merged_dict_1.update(_)
for _ in gg_dict:
    merged_dict_2.update(_)
for _ in rnd_dict:
    merged_dict_3.update(_)

ccp_idf = compute_idf([merged_dict_3, merged_dict_1, merged_dict_2])

ccp_tfidf = zip_tfidf(ccp_tf, ccp_idf)
gg_tfidf = zip_tfidf(gg_tf, ccp_idf)
rnd_tfidf = zip_tfidf(rnd_tf, ccp_idf)


def get_tfidf_result(tfidf_zipped):
    result_tfidf = {}
    for dicts in tfidf_zipped:
        result_tfidf.update(dicts)
    return result_tfidf


result_ccp = get_tfidf_result(ccp_tfidf)
result_gg = get_tfidf_result(gg_tfidf)
result_rnd = get_tfidf_result(rnd_tfidf)

for key, value in result_ccp.items():
    if value < 0.00001:
        stop_words.append(key)

for key, value in result_gg.items():
    if value < 0.00001:
        stop_words.append(key)

for key, value in result_rnd.items():
    if value < 0.00001:
        stop_words.append(key)

with open("stopwords.txt", "w+") as stopword_file:
    for stopwords in stop_words:
        stopword_file.write(f"{stopwords}\n")


def sort_and_write_words(tfidf_dict, name):
    tfidf_output = dict(sorted(tfidf_dict.items(), key=lambda key: key[1], reverse=True))
    counter_tfidf = Counter(tfidf_output)
    words_to_write = counter_tfidf.most_common(20)
    with open(f"{name}.txt", "w+") as f:
        for items in words_to_write:
            f.write(f"{items[0]}\n")


sort_and_write_words(result_ccp, "ccp")
sort_and_write_words(result_gg, "gg")
sort_and_write_words(result_rnd, "rnd")
