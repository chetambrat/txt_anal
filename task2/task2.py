import gensim


def memoize(func):
    mem = {}

    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]

    return memoizer


@memoize
def levenshtein(first_word, second_word):
    """ reference: https://www.python-course.eu/levenshtein_distance.php """
    if first_word == "":
        return len(second_word)
    if second_word == "":
        return len(first_word)
    if first_word[-1] == second_word[-1]:
        cost = 0
    else:
        cost = 1

    res = min([levenshtein(first_word[:-1], second_word) + 1,
               levenshtein(first_word, second_word[:-1]) + 1,
               levenshtein(first_word[:-1], second_word[:-1]) + cost])

    return res


def hamming(first_word, second_word):
    """ use only with len(a) == len(b) """
    if len(first_word) == len(second_word):
        return sum(byte1 != byte2 for byte1, byte2 in zip(first_word, second_word))
    else:
        return abs(len(second_word) - len(first_word))


def jaccard(first_word, second_word):
    word1 = set(first_word)
    word2 = set(second_word)
    return float(len(word1.intersection(word2))) / len(word1.union(word2))


def dam_lev_d(first_word, second_word):
    """ ref: wiki """
    d = {}
    lenstr1 = len(first_word)
    lenstr2 = len(second_word)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if first_word[i] == second_word[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,
                d[(i, j - 1)] + 1,
                d[(i - 1, j - 1)] + cost,
            )
            if i and j and first_word[i] == second_word[j - 1] and first_word[i - 1] == \
                    second_word[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)

    return d[lenstr1 - 1, lenstr2 - 1]


class Words:
    def __init__(self, first_word, second_word):
        self.first_word = first_word
        self.second_word = second_word

    def result(self):
        return print(
            f"for words {self.first_word, self.second_word} \n hamming distance is: {hamming(self.first_word, self.second_word)} "
            f"\n levenstein distance is: {levenshtein(self.first_word, self.second_word)} \n jaccard metric is:"
            f" {jaccard(self.first_word, self.second_word)} "
            f"\n dam lev distance is"
            f" {dam_lev_d(self.first_word, self.second_word)}")


example1 = Words("abc", "dbc")
example2 = Words("abc", "a")


example1.result()
example2.result()
