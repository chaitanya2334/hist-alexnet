import glob
import os
from collections import defaultdict, namedtuple
from random import shuffle

import cv2
import itertools
from torch.utils import data
from tqdm import tqdm
import csv_utils
import numpy as np

from preprocessing import tiler
from preprocessing.normal_staining import normalize_staining
# from preprocessing.tiler import rotate, tile, vert_reflect
from utils import fit_image

Sample = namedtuple("Sample", ["image", "label", "i_id", "t_id"])


class BIC_Dataset(data.Dataset):
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

    def __init__(self, train_dir, test_dir, train_save_dir, test_save_dir, label_2_id, mu, std, tile_size=299, stride=150,
                 dev_per=20, order=None, randomize=True, transform=None, redo_preprocessing=False):

        self.dir_status_file = ".dir_status"
        self.tile_size = tile_size
        self.stride = stride

        self.randomize = randomize

        self.train_dir = train_dir
        self.test_dir = test_dir

        if transform == 'normalize':
            self.transform = self.__normalize

        self.mu = mu
        self.std = std

        # dict [benign:0, normal:1 ...]
        self.label_2_id = label_2_id
        # get labels ids from csv file
        self.train_lbls = csv_utils.read(os.path.join(train_dir, "labels.csv"))
        self.test_lbls = csv_utils.read(os.path.join(test_dir, "labels.csv"))

        self.dispatch = {
            'tile': self.__tile,
            'normal_stain': self.__normal_stain,
        }

        if order is None:
            order = ['normal_stain', 'tile']

        self.__preprocess_images(train_dir, train_save_dir, order, redo_preprocessing)
        self.__preprocess_images(test_dir, test_save_dir, order, redo_preprocessing)

        self.__create_dataset(train_save_dir, test_save_dir, dev_per)

    def __preprocess_images(self, src_dir, dest_dir, order, redo):
        # run based on the given order in functions
        src_files = glob.glob(src_dir + "/*.tif")
        dir_name = os.path.basename(os.path.normpath(src_dir))
        t = tqdm(src_files)

        for img_file in t:
            file_name = os.path.splitext(os.path.basename(img_file))[0]
            dest_path = os.path.join(dest_dir, file_name)
            t.desc = "Preprocessing images \"{0}\" in directory \"{1}\" ".format(file_name, dir_name)

            if not self.__do_images_exist(dest_path) or redo:
                img = cv2.imread(img_file)
                img = img.astype(np.float64, copy=False)

                if img is None:
                    print(img_file)

                for func in order:
                    img = self.dispatch[func](img)

                self.__save_images(dest_path, img)

    def __do_images_exist(self, in_dir):
        # various checks to make sure that in_dir contains all the tiles
        # generated in the previous run (that we may want to reuse)
        try:
            f = open(os.path.join(in_dir, self.dir_status_file), 'r')
            status = int(f.read())
            return status == 1
        except FileNotFoundError:
            return False

    def __save_images(self, save_path, imgs):

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        try:
            if isinstance(imgs, list):
                for i, img in enumerate(imgs):
                    cv2.imwrite(os.path.join(save_path, str(i) + ".png"), img)

            elif isinstance(imgs, np.ndarray):
                cv2.imwrite(save_path + ".png", imgs)

            else:
                raise TypeError("imgs is of type{0}".format(type(imgs)))

            # everything went well, record positive status
            with open(os.path.join(save_path, self.dir_status_file), 'w') as f:
                f.write("1")

        except KeyboardInterrupt:
            # interrupted, record negative status
            with open(os.path.join(save_path, self.dir_status_file), 'w') as f:
                f.write("0")

    def __create_dataset(self, train_dir, test_dir, dev_per):

        # get image metadata for each image in the training and test directories
        self.train_desc = self.__get_tile_desc(train_dir, self.train_lbls, "tr")
        self.test_desc = self.__get_tile_desc(test_dir, self.test_lbls, "te")

        # split train and dev set from all images in training directory
        # such that percentage of each label is maintained in both the sets.
        self.train_desc, self.dev_desc = self.__uniform_split(self.train_desc, size2=dev_per)

        # create a Dataset object for each train, dev and test.
        self.train = self.CustomDataset(self.train_desc, self.label_2_id, self.transform)
        self.dev = self.CustomDataset(self.dev_desc, self.label_2_id, self.transform)
        self.test = self.CustomDataset(self.test_desc, self.label_2_id, self.transform)

    def __normal_stain(self, img):
        if isinstance(img, list):
            res = []
            for i in img:
                res.append(self.__normal_stain(i))

            return res

        if isinstance(img, np.ndarray):
            return normalize_staining(img)

    def __normalize(self, img):

        img = img.astype(np.float64, copy=False)
        for ch in range(img.shape[2]):
            img[:, :, ch] = (img[:, :, ch] - self.mu[ch]) / self.std[ch]

        return img

    def __tile(self, img):

        images = tile(img, size=self.tile_size, stride=self.stride)
        rot_images = rotate(images)
        ref_images = vert_reflect(rot_images)
        return ref_images

    def __uniform_split(self, desc, size2):
        def split(l, size):
            ids = list(set([int(i[2][2:]) for i in l]))
            if size == 0:
                id1 = ids[:]
                id2 = []
            else:
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
            label_only_set = [item for item in desc if item[1].lower() == label.lower()]
            total_imgs = list(set([int(i[2][2:]) for i in label_only_set]))
            t, d = split(label_only_set, int(len(total_imgs) * (size2 / 100)))
            train_desc.extend(t)
            dev_desc.extend(d)
        print(len(train_desc), len(dev_desc), total_len)

        assert len(train_desc) + len(dev_desc) == total_len
        return train_desc, dev_desc

    @staticmethod
    def _label_by_id(labels, img_id):

        for row in labels:
            if row['id'] == img_id:
                return row['id'], row['label']

        return None, None

    def __get_tile_desc(self, path, labels, id_append):
        """
        generates a list of tuples (image_path, label, image_id) for every whole slide image.

        :param path: the path to images for which you want to generate the above tuple
        :type path: str
        :param labels: a list of dictionaries extracted from reading the labels csv file
        :type labels: list[dict[str, str]]
        :param id_append: a string to be appended to the image_id
        :type id_append: str
        :return: list(tuple())
        :rtype: list[(str, str, str)]
        """
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

    def batchify(self, samples):
        img_size = self.tile_size
        images = [fit_image(sample.image, img_size) for sample in samples if sample]
        images = np.stack(images, axis=0)
        labels = [sample.label for sample in samples if sample]
        i_ids = [sample.i_id for sample in samples if sample]
        t_ids = [sample.t_id for sample in samples if sample]
        return images, labels, i_ids, t_ids
