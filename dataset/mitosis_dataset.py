import glob
import os
from collections import defaultdict, namedtuple
from random import shuffle

import cv2
import itertools

import torch
from torch.utils import data


Sample = namedtuple("Sample", ["image", "label"])


class MitosisDataset(data.Dataset):
    class CustomDataset(data.Dataset):
        def __init__(self, path_label_pair, label_2_id, transform=None):
            self.path_label_pair = path_label_pair
            self.transform = transform
            self.label_2_id = label_2_id

        def __getitem__(self, item):
            filepath, lbl_idx = self.path_label_pair[item]
            image = cv2.imread(filepath)
            lbl_idx = self.label_2_id[lbl_idx]
            if self.transform is not None:
                image = self.transform(image)

            return Sample(image, lbl_idx)

        def __len__(self):
            return len(self.path_label_pair)

    def __init__(self, mito_dir, no_mito_dir, label_2_id,
                 split=(90, 5, 5), randomize=True, transform=None):

        self.dir_status_file = ".dir_status"

        self.randomize = randomize

        self.mito_dir = mito_dir
        self.no_mito_dir = no_mito_dir

        # dict [benign:0, normal:1 ...]
        self.label_2_id = label_2_id

        # dict {"mitosis" : [paths],
        #       "no-mitosis": [paths]}
        if os.path.isdir(glob.glob(os.path.join(self.mito_dir, "*"))[0]):
            full_data = {"mitosis": sorted(glob.glob(os.path.join(self.mito_dir, "*", "*.png")), key=os.path.getmtime),
                         "no-mitosis": sorted(glob.glob(os.path.join(self.no_mito_dir, "*", "*.png")),
                                              key=os.path.getmtime)}

        else:
            full_data = {"mitosis": sorted(glob.glob(os.path.join(self.mito_dir, "*.png")), key=os.path.getmtime),
                         "no-mitosis": sorted(glob.glob(os.path.join(self.no_mito_dir, "*.png")), key=os.path.getmtime)}

        self.transform = transform

        self.__create_dataset(split, full_data)

    @staticmethod
    def batchify(samples):
        images = [sample.image for sample in samples if sample]
        images = torch.stack(images, dim=0)
        labels = [sample.label for sample in samples if sample]
        return images, labels

    @staticmethod
    def __split(per, bins):
        res = ([], [], [])
        assert sum(per) == 100
        print(res)
        for k, l in bins.items():
            size = len(l)
            print("label: {0}, size: {1}".format(k, len(l)))
            cum_per = 0
            prv = 0
            for i, p in enumerate(per):
                cum_per += p
                nxt = int((cum_per / 100) * size)
                res[i].extend([(val, k) for val in l[prv:nxt]])
                print(len(res[0]), len(res[1]), len(res[2]))
                prv = nxt

        return res

    def __create_dataset(self, split, full_dataset):

        # get image metadata for each image in the training and test directories
        tr, dev, te = self.__split(split, full_dataset)

        # create a Dataset object for each train, dev and test.
        self.train = self.CustomDataset(tr, self.label_2_id, self.transform)
        self.dev = self.CustomDataset(dev, self.label_2_id, self.transform)
        self.test = self.CustomDataset(te, self.label_2_id, self.transform)

    @staticmethod
    def _label_by_id(labels, img_id):

        for row in labels:
            if row['id'] == img_id:
                return row['id'], row['label']

        return None, None
