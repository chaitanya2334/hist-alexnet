import glob
import os
from collections import defaultdict, namedtuple
from random import shuffle

import cv2
import itertools
from torch.utils import data
from tqdm import tqdm
import csv_utils

Sample = namedtuple("Sample", ["image", "label", "i_id", "t_id"])


class BIC_tiled_Dataset(data.Dataset):
    class CustomDataset(data.Dataset):
        def __init__(self, tile_desc, label_2_id, transform=None):
            self.tile_desc = tile_desc
            self.transform = transform
            self.label_2_id = label_2_id

        def __getitem__(self, item):
            filepath, label, i_id, t_id = self.tile_desc[item]
            image = cv2.imread(filepath)
            label = self.label_2_id[label]
            if self.transform is not None:
                image = self.transform(image)

            return Sample(image, label, i_id, t_id)

        def __len__(self):
            return len(self.tile_desc)

    def __init__(self, train_dir, test_dir, label_2_id, randomize=True, transform=None):
        self.randomize = randomize
        self.train_lbls = csv_utils.read(os.path.join(train_dir, "labels.csv"))
        self.test_lbls = csv_utils.read(os.path.join(test_dir, "labels.csv"))
        self.label_2_id = label_2_id

        train_desc = self.__get_tile_desc(train_dir, self.train_lbls, "tr")
        train_desc, dev_desc = self.split_train_dev(train_desc, dev_size=35)
        test_desc = self.__get_tile_desc(test_dir, self.test_lbls, "te")

        self.train = self.CustomDataset(train_desc, label_2_id, transform)
        self.dev = self.CustomDataset(dev_desc, label_2_id, transform)
        self.test = self.CustomDataset(test_desc, label_2_id, transform)

    def split_train_dev(self, desc, dev_size):
        def split(l, size):
            ids = list(set([int(i[2][2:]) for i in l]))
            id1 = ids[:-size]
            id2 = ids[-size:]
            print(len(id1), len(id2))
            l1 = [i for i in l if int(i[2][2:]) in id1]
            l2 = [i for i in l if int(i[2][2:]) in id2]
            return l1, l2

        total_len = len(desc)

        train_desc = []
        dev_desc = []
        sorted(desc, key=lambda element: element[2])

        for label, v in self.label_2_id.items():
            label_only_set = [item for item in desc if item[1] == label]
            t, d = split(label_only_set, int(dev_size/len(self.label_2_id)))
            train_desc.extend(t)
            dev_desc.extend(d)
        print(len(train_desc), len(dev_desc), total_len)
        assert len(train_desc) + len(dev_desc) == total_len
        return train_desc, dev_desc

    @staticmethod
    def __split_dataset(per, bins):
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
                res[i].extend(l[prv:nxt])
                print(len(res[0]), len(res[1]), len(res[2]))
                prv = nxt

        return res

    @staticmethod
    def _label_by_id(labels, img_id):

        for row in labels:
            if row['id'] == img_id:
                return row['id'], row['label']

        return None, None

    def __get_tile_desc(self, path, labels, id_append):
        tile_desc = []
        paths = glob.glob(os.path.join(path, "*"))
        if self.randomize:
            shuffle(paths)
        for wspath in tqdm(paths, desc="Reading Image Folders for {0}".format(path)):
            img_id = id_append + os.path.basename(os.path.normpath(wspath))
            filepaths = glob.glob(os.path.join(wspath, "*.png"))
            for filepath in filepaths:
                tile_id = os.path.splitext(os.path.basename(filepath))[0]  # just the filename (remove extension)

                _, label = self._label_by_id(labels, img_id[2:])

                tile_desc.append((filepath, label, img_id, tile_id))

        return tile_desc

    def __bin_paths(self, wspaths):
        d = defaultdict(list)
        for wspath in wspaths:
            wsid = os.path.splitext(os.path.basename(wspath))[0]
            _, label = self._label_by_id(wsid)
            if label:
                label = self.label2id[label]
                d[label].append(wspath)

        return d

    @staticmethod
    def __print_dataset(datasets, names, labels):
        for dataset, name in zip(datasets, names):
            print(name)
            for label in labels:
                print("{0} : {1}".format(label, sum([item[1] == label for item in dataset])))

    def __read_and_split(self, image_dir, split, randomize):
        # assumes the following file structure for access to images:
        # folder
        #   |
        #   ├── TCGA....
        #   |   |
        #   |   ├── label
        #   |   ├── macro
        #   |   ├── thumbnail
        #   |   ├── slide
        #   |   |   |
        #   |   |   ├── (int)
        #   |   |   |   |
        #   |   |   |   ├── <x1>_<y1>.jpeg
        #   |   |   |   ├── <x2>_<y2>.jpeg
        #   |   |   |   ├── <x3>_<y3>.jpeg
        #   ├── TCGA ....

        wspaths = sorted(glob.glob(image_dir + "/*"))

        binned_wspaths = self.__bin_paths(wspaths)

        if randomize:
            shuffle(wspaths)

        print(len(wspaths))
        wspaths_train, wspaths_dev, wspaths_test = self.__split_dataset(split, binned_wspaths)

        print(len(wspaths_train), len(wspaths_dev), len(wspaths_test))

        metadata_train = self.__read_metadata("train", wspaths_train)
        metadata_dev = self.__read_metadata("dev", wspaths_dev)
        metadata_test = self.__read_metadata("test", wspaths_test)

        self.__print_dataset((metadata_train, metadata_dev, metadata_test), ("train", "dev", "test"),
                             binned_wspaths.keys())

        return metadata_train, metadata_dev, metadata_test

    def __read(self, image_dir, randomize):
        wspaths = sorted(glob.glob(image_dir + "/*"))
        if randomize:
            shuffle(wspaths)

        metadata = self.__read_metadata("train", wspaths)
        return metadata

    def __all_labels(self, name):
        return [row[name] for row in self.__labels_csv]
