import glob
import imghdr
import os
import torch
from collections import namedtuple, defaultdict, Counter
from random import shuffle

import cv2
from numpy.linalg import LinAlgError
from torch.utils import data
from tqdm import tqdm

from preprocessing.normal_staining import normalize_staining
import numpy as np

Sample = namedtuple('Sample', ['image', 'label', "wsid", "x", "y"])


class CustomDataset(data.Dataset):
    def __init__(self, metadata, label2id, transform=None):
        self.__metadata = metadata
        self.label2id = label2id
        self.transform = transform

    def get_sample_weights(self):
        labels = [label for _, label, _, _, _ in self.__metadata]
        c = Counter(labels)
        weights = list(c.values())
        weights = [1.0 / c for c in weights]
        sample_weights = [weights[label] for _, label, _, _, _ in self.__metadata]
        return sample_weights

    def __getitem__(self, item):
        filepath, label, wsid, x, y = self.__metadata[item]
        image = cv2.imread(filepath)
        # image = self.__preprocess_images(filepath)

        if self.transform is not None:
            image = self.transform(image)

        return Sample(image, label, wsid, x, y)

    @staticmethod
    def __do_images_exist(file_path):
        # various checks to make sure that in_dir contains all the tiles
        # generated in the previous run (that we may want to reuse)
        try:
            ext = imghdr.what(file_path)
            return ext == "jpeg"
        except FileNotFoundError:
            return False

    def __preprocess_images(self, src_file):
        # run based on the given order in functions

        img = cv2.imread(src_file)
        img = img.astype(np.float64, copy=False)

        if img is None:
            print("NO IMAGE IN FILE: {0}".format(src_file))

        img = self.__normal_stain(img)

        return img

    @staticmethod
    def __save_images(save_path, imgs):

        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if isinstance(imgs, list):
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(save_path, str(i)), img)

        elif isinstance(imgs, np.ndarray):
            cv2.imwrite(save_path, imgs)

        else:
            raise TypeError("imgs is of type{0}".format(type(imgs)))

    def __normal_stain(self, img):
        if isinstance(img, list):
            res = []
            for i in img:
                res.append(self.__normal_stain(i))

            return res

        if isinstance(img, np.ndarray):
            try:
                ret = normalize_staining(img)
            except LinAlgError as e:
                ret = img
                print(e)
            return ret

    def __len__(self):
        return len(self.__metadata)


class OCDataset(object):
    def __init__(self, onco_dir, chromo_dir, onco_norm_dir, chromo_norm_dir, split, label2id=None, randomize=False,
                 transform=None, redo_preprocessing=False):

        self.label2id = label2id
        self.dir_status_file = ".dir_status"

        self.__preprocess_images(onco_dir, onco_norm_dir, redo_preprocessing)
        self.__preprocess_images(chromo_dir, chromo_norm_dir, redo_preprocessing)

        onco_train, onco_dev, onco_test = self.__read_folders(onco_norm_dir, split, randomize, label="oncocytoma")
        chromo_train, chromo_dev, chromo_test = self.__read_folders(chromo_norm_dir, split, randomize,
                                                                    label="chromophobe")

        self.__metadata_train = onco_train + chromo_train
        self.__metadata_dev = onco_dev + chromo_dev
        self.__metadata_test = onco_test + chromo_test

        self.__print_dataset((self.__metadata_train, self.__metadata_dev, self.__metadata_test),
                             ("train", "dev", "test"),
                             self.label2id.values())

        self.train = CustomDataset(self.__metadata_train, self.label2id, transform)
        self.dev = CustomDataset(self.__metadata_dev, self.label2id, transform)
        self.test = CustomDataset(self.__metadata_test, self.label2id, transform)

        print("total training examples: {0}".format(len(self.train)))
        print("total dev samples: {0}".format(len(self.dev)))
        print("total test samples: {0}".format(len(self.test)))

        assert len(self.train) + len(self.dev) + len(self.test) == len(self.__metadata_train) + len(
            self.__metadata_dev) + len(self.__metadata_test)

    @staticmethod
    def __split_dataset(per, bins):
        res = ([], [], [])
        assert sum(per) == 100
        for k, l in bins.items():
            size = len(l)
            print("label: {0}, size: {1}".format(k, len(l)))
            cum_per = 0
            prv = 0
            for i, p in enumerate(per):
                cum_per += p
                nxt = int((cum_per / 100) * size)
                res[i].extend(l[prv:nxt])
                prv = nxt

        print("split -- train: {0}, dev: {1}, test: {2}".format(len(res[0]), len(res[1]), len(res[2])))
        return res

    def __read_metadata(self, name, wspaths, label):
        metadata = []
        for wspath in tqdm(wspaths, desc="Reading Image Folders for {0}".format(name)):
            wsid = os.path.splitext(os.path.basename(wspath))[0]
            filepaths = glob.glob(os.path.join(wspath, "slide", "*", "*.jpeg"))
            for filepath in filepaths:
                basename = os.path.splitext(os.path.basename(filepath))[0]  # just the filename (remove extension)

                pos = basename.split("_")

                if len(pos) == 2:
                    x, y = pos
                elif len(pos) == 3:
                    x, y, a = pos
                else:
                    x = pos[0]
                    y = pos[1]

                l_id = self.label2id[label]

                metadata.append((filepath, l_id, wsid, x, y))

        return metadata

    def __preprocess_images(self, src_dir, dest_dir, redo):
        # run based on the given order in functions
        src_files = list(glob.iglob(src_dir + '/**/*.jpeg', recursive=True))
        dir_name = os.path.basename(os.path.normpath(src_dir))
        t = tqdm(src_files)

        for src_file in t:
            dest_file = src_file
            dest_file = dest_file.replace(src_dir, dest_dir)

            t.desc = "Preprocessing images \"{0}\" in directory \"{1}\" ".format(os.path.basename(src_file), dir_name)
            if not self.__do_images_exist(dest_file) or redo:
                img = cv2.imread(src_file)
                img = img.astype(np.float64, copy=False)

                if img is None:
                    print("NO IMAGE IN FILE: {0}".format(src_file))

                img = self.__normal_stain(img)
                self.__save_images(dest_file, img)

    @staticmethod
    def __do_images_exist(file_path):
        # various checks to make sure that in_dir contains all the tiles
        # generated in the previous run (that we may want to reuse)
        try:
            ext = imghdr.what(file_path)
            return ext == "jpeg"
        except FileNotFoundError:
            return False

    @staticmethod
    def __save_images(save_path, imgs):

        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if isinstance(imgs, list):
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(save_path, str(i)), img)

        elif isinstance(imgs, np.ndarray):
            cv2.imwrite(save_path, imgs)

        else:
            raise TypeError("imgs is of type{0}".format(type(imgs)))

    def __normal_stain(self, img):
        if isinstance(img, list):
            res = []
            for i in img:
                res.append(self.__normal_stain(i))

            return res

        if isinstance(img, np.ndarray):
            try:
                ret = normalize_staining(img)
            except LinAlgError as e:
                ret = img
                print(e)
            return ret

    @staticmethod
    def batchify(samples):
        images = [sample.image for sample in samples if sample]
        images = torch.stack(images, dim=0)
        labels = [sample.label for sample in samples if sample]
        return images, labels

    def __bin_paths(self, wspaths, label):
        d = defaultdict(list)
        for wspath in wspaths:
            l_id = self.label2id[label]
            d[l_id].append(wspath)

        return d

    @staticmethod
    def __print_dataset(datasets, names, labels):
        for dataset, name in zip(datasets, names):
            print(name)
            for label in labels:
                print("{0} : {1}".format(label, sum([item[1] == label for item in dataset])))

    def __read_folders(self, image_dir, split, randomize, label):
        # assumes the following file structure for access to images:
        # folder
        #   |
        #   ├── Case....
        #   |   |
        #   |   ├── label
        #   |   ├── macro
        #   |   ├── thumbnail
        #   |   ├── slide
        #   |   |   |
        #   |   |   ├── 16
        #   |   |   |   |
        #   |   |   |   ├── <x1>_<y1>.jpeg
        #   |   |   |   ├── <x2>_<y2>.jpeg
        #   |   |   |   ├── <x3>_<y3>.jpeg
        #   ├── Case ....

        wspaths = sorted(glob.glob(image_dir + "/*"))

        binned_wspaths = self.__bin_paths(wspaths, label)

        if randomize:
            shuffle(wspaths)

        wspaths_train, wspaths_dev, wspaths_test = self.__split_dataset(split, binned_wspaths)

        metadata_train = self.__read_metadata("train", wspaths_train, label)
        metadata_dev = self.__read_metadata("dev", wspaths_dev, label)

        metadata_test = self.__read_metadata("test", wspaths_test, label)

        return metadata_train, metadata_dev, metadata_test
