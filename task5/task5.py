from math import log

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

path = "spam.csv"
stemming = PorterStemmer()

def process_data(location):
    mails = pd.read_csv(location, encoding='latin-1')
    mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    mails.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
    mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})
    mails.drop(['labels'], axis=1, inplace=True)
    return mails


def split_train_test(dataset):
    train_splitting, test_splitting = [], []
    for splitting_index in range(dataset.shape[0]):
        if np.random.uniform(0, 1) < 0.80:
            train_splitting += [splitting_index]
        else:
            test_splitting += [splitting_index]
    train = dataset.loc[train_splitting]
    test = dataset.loc[test_splitting]
    train.reset_index(inplace=True)
    train.drop(['index'], axis=1, inplace=True)

    test.reset_index(inplace=True)
    test.drop(['index'], axis=1, inplace=True)
    return train, test


mail_dataset = process_data(path)
train_data, test_data = split_train_test(mail_dataset)


def process_message(message):
    message = message.lower()
    processed_words = word_tokenize(message)
    processed_words = [w for w in processed_words if len(w) > 2]
    processed_words = [w for w in processed_words if w not in stopwords.words('english')]
    processed_words = [stemming.stem(w) for w in processed_words]
    return processed_words


class NaiveBayesClassifier:
    # ref: https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier
    def __init__(self, train_data):
        self.mails, self.labels = train_data['message'], train_data['label']
        self.sum_tf_idf_ham, self.sum_tf_idf_spam, self.ham_words, self.spam_words = 0, 0, 0, 0
        self.prob_ham, self.prob_spam, self.idf_ham, self.tf_ham, self.idf_spam, self.tf_spam = {}, {}, {}, {}, {}, {}
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def train(self):
        self.calculate_tf_idf()

    def calc_prob(self):
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words +
                                                             len(list(self.tf_ham.keys())))

    def calculate_tf_idf(self):
        self.prob_spam = {}
        self.prob_ham = {}
        number_of_msg = self.mails.shape[0]
        for i in range(number_of_msg):
            processed_message = process_message(self.mails[i])
            count = []
            for word in processed_message:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log(
                (self.spam_mails + self.ham_mails) / (self.idf_spam[word] +
                                                      self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
            self.prob_spam[word] = (self.prob_spam[word] + 1) / \
                                   (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails)
                                                            / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))

    def classify(self, processed_message):
        count_spam, count_ham = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                count_spam += log(self.prob_spam[word])
            else:
                count_spam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

            if word in self.prob_ham:
                count_ham += log(self.prob_ham[word])
            else:
                count_ham -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))

            count_spam += log(self.prob_spam_mail)
            count_ham += log(self.prob_ham_mail)
        return count_spam >= count_ham

    def predict(self, test_data):
        result = {}
        for (message_iterator, message) in enumerate(test_data):
            processed_message = process_message(message)
            result[message_iterator] = int(self.classify(processed_message))
        return result


def calculate_metrics(labels, predicts):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(labels)):
        tp += int(labels[i] == 1 and predicts[i] == 1)
        tn += int(labels[i] == 0 and predicts[i] == 0)
        fp += int(labels[i] == 0 and predicts[i] == 1)
        fn += int(labels[i] == 1 and predicts[i] == 0)
    pr = tp / (tp + fp)
    rc = tp / (tp + fn)
    fsc = 2 * pr * rc / (pr + rc)
    acc = (tp + tn) / (tp + tn + fp + fn)

    with open("metrics.txt", "w") as metrics_file:
        metrics_file.write(f"Precision: {pr}\nRecall: {rc}\nFscore: {fsc}\nAccuracy: {acc}")


classifier = NaiveBayesClassifier(train_data)
classifier.train()
predictions = classifier.predict(test_data['message'])
calculate_metrics(test_data['label'], predictions)

spam = process_message('Win free 100$ by answering survey')
assert classifier.classify(spam)

not_spam = process_message('Do you feel like a hero yet?')
assert classifier.classify(not_spam)
