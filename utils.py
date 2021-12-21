#!/usr/bin/python
# encoding: utf-8

import collections
import numpy as np
from Levenshtein import distance


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """encode batch str.

        Args:
            text (list of str): texts to convert.

        Returns:
            np.array [n, l]: encoded texts. (l = Maximum of all text lengths)
            np.array [n]:    length of each text.

        Example:
            >>> encode(["Aa", "abc"])
            np.array([[11, 11, 0], [11, 12, 13]]), np.array([2, 3])
        """

        length = [len(t) for t in text]
        max_length = max(length)
        text = [[self.dict[char.lower() if self._ignore_case else char] for char in t] + [0] * (max_length - len(t)) for t in text]
        return np.array(text), np.array(length)

    def decode(self, encoded_t, raw=False):
        """Decode encoded texts back into strs.

        Args:
            encoded_t (np.array [n, length]): encoded text to decode
            raw       (bool): do not ignore '-' and repeated characters

        Returns:
            list of decoded texts
        
        Example:
            >>> decode(np.array([[11, 11, 0], [11, 12, 13]]))
            raw=True:  ["aa-", "abc"]
            raw=False: ["a", "abc"]
        """   

        texts = []
        length = encoded_t.shape[1]
        for t in encoded_t:
            if raw:
                texts.append(''.join([self.alphabet[i - 1] for i in t]))
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                texts.append(''.join(char_list))
        return texts



class averager(object):
    """Compute average for numpy.array."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.size
        v = v.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class BKTreeNode(object):
    def __init__(self, s):
        self.s = s
        self.son = {}

    def insert(self, s):
        dis = distance(self.s, s)
        if dis in self.son.keys():
            self.son[dis].insert(s)
        else:
            self.son[dis] = BKTreeNode(s)

    def find(self, s, threshold, ret_list):
        dis = distance(self.s, s)
        if dis <= threshold:
            ret_list.append(self.s)

        for key, value in self.son.items():
            if dis - threshold <= key <= dis + threshold:
                value.find(s, threshold, ret_list)


class BKTree(object):
    def __init__(self, lex_list):
        for i, lex in enumerate(lex_list):
            if i == 0:
                self.root = BKTreeNode(lex)
            else:
                self.root.insert(lex)

    def find(self, s, threshold):
        ret_list = []
        self.root.find(s, threshold, ret_list)
        return ret_list


def array2str(arr, separator=", "):
    """
    transform Iterable to string. 
    e.g. [[1, 2], [3]]  ->  "[[1, 2], [3]]"
    """
    if isinstance(arr, collections.Iterable):
        return "[" + separator.join([array2str(x) for x in arr]) + "]"
    else:
        return str(arr)

def jtArray2str(arr):
    """
    transform jt.array to string. 
    """
    return array2str(np.array(arr).tolist())


