#!/usr/bin/python
# encoding: utf-8

import random
import jittor as jt
from jittor.dataset.dataset import Dataset
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import os

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, batch_size=1, shuffle=False, num_workers=0):
        super(lmdbDataset, self).__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.transform = transform
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.total_len = int(txn.get('num-samples'.encode()))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle, num_workers=num_workers)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode()

        return (img, label)



class Synth90kDataset(Dataset):

    def __init__(self, root="", mode="train", transform=None, batch_size=1, shuffle=False, num_workers=0):
        super(Synth90kDataset, self).__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        mapping = {}
        with open(os.path.join(root, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()
        paths = []
        texts = []
        with open(os.path.join(root, "annotation_{}.txt".format(mode)), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        self.paths = paths
        self.texts = texts
        self.total_len = len(self.paths)

        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle, num_workers=num_workers)
        print("{} dataset loaded".format(mode))

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            img = Image.open(path).convert('L')  # grey-scale
            img = self.transform(img)
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        return (img, self.texts[index])




class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = (np.array(img) / 127.5) - 1.0
        return np.expand_dims(img, axis=0).astype(np.float32)
