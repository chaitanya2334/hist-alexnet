import glob
import math
import os
from itertools import compress, chain
from sys import stderr

import torch
from collections import namedtuple, defaultdict, Counter
from random import shuffle

import cv2
from Cython.Utils import OrderedSet
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm import tqdm
import csv_utils
from model.alexnet_mitoses import alexnet
from preprocessing.normal_staining import normalize_staining
from utils import argmax, prob

Sample = namedtuple('Sample', ['image', 'label', "wsid", "x", "y"])


class CustomDataset(data.Dataset):
    def __init__(self, wsid, metadata, label2id, transform=None):
        self.__metadata = metadata
        self.label2id = label2id
        self.transform = transform
        self.wsid = wsid

    def get_sample_weights(self, uniform=False):
        labels = [label for _, label, _, _, _ in self.__metadata]
        c = Counter(labels)
        if not uniform:
            weights = {k: (1 / v) * 1000 for k, v in c.items()}
        else:
            weights = {k: 1 / len(c.keys()) for k, v in c.items()}
        return weights, labels

    def __getitem__(self, item):
        try:
            filepath, label, wsid, x, y = self.__metadata[item]
        except IndexError as e:
            print(e, file=stderr)
            print(self.wsid)

        image = cv2.imread(filepath)

        if self.transform is not None:
            image = self.transform(image)

        return Sample(image, label, wsid, x, y)

    def __len__(self):
        return len(self.__metadata)


class TcgaWsiDataset(object):
    def __init__(self,
                 class_type,
                 image_dir,
                 label_filepath,
                 split,
                 label2id=None,
                 transform=None,
                 filter_model=None,
                 filter_percent=100):

        self.class_type = class_type
        self.__labels_csv = csv_utils.read(label_filepath)

        if label2id:
            self.label2id = label2id
            print(self.label2id)
        else:
            unique = list(OrderedSet(self.__all_labels(self.class_type)))
            self.label2id = dict(zip(unique, range(len(unique))))
            print(self.label2id)

        self.n_labels = len(self.label2id.keys())
        self.train, self.dev, self.test = self.gen_triples(image_dir, split, transform, filter_model, filter_percent)

    def gen_triples(self, image_dir, split, transform, filter_model, filter_percent):
        # Step 1. first read all the tiles and then split into train dev and test evenly.
        # such that the distribution of labels is the same.
        # t_train is a dict where the key is wsipath and value is a list of tile metadata
        t_train, t_dev, t_test = \
            self.read_tiles_and_split(image_dir, split)

        # Step 2. filter the tiles based on whatever filter we choose
        if filter_model and filter_percent < 100:
            t_train, t_dev, t_test = \
                (self.mitoses_filter(t, filter_model, per=filter_percent) for t in (t_train, t_dev, t_test))

        # Step 3. encapsulate the tiles in each wsi in CustomDataset containers
        # to allow easy processing through the torch pipeline
        train, dev, test = \
            ([CustomDataset(wsid, wsi_metadata, self.label2id, transform) for wsid, wsi_metadata in d.items()]
             for d in (t_train, t_dev, t_test))

        return train, dev, test

    def mitoses_filter(self, tiles, model_fp, per=25):
        # prepare dataset of tiles
        transform = Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        t_dataset = CustomDataset(tiles, self.label2id, transform=transform)
        data_loader = DataLoader(dataset=t_dataset,
                                 batch_size=64,
                                 num_workers=28,
                                 collate_fn=self.batchify)

        # load model
        model = alexnet(pretrained=True, num_classes=2)
        model = model.cuda()
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['state_dict'])
        model.training = False

        # infer every tile's label using trained mitoses model.
        # get prob for tile being mitotic
        def infer(images):
            mitoses = 0
            inp = Variable(images.cuda())
            seq_out = model(inp)
            return prob(seq_out, label=mitoses)

        t_probs = []
        t_wsids = []
        # a list of probabilities for each tile (P(T = mitoses))
        for images, _, img_paths in tqdm(data_loader, desc="mitoses filtering"):
            probs = infer(images)
            t_probs.extend(probs)
            t_wsids.extend(img_paths)

        # gen a dictionary of top n tiles indexed by wsid
        top_n = defaultdict(list)
        for wsid, p in zip(t_wsids, t_probs):
            top_n[wsid].append(p)

        # sort and then crop to get top n
        for k in top_n.keys():
            n = math.ceil(len(top_n[k]) * (per / 100))
            top_n[k] = sorted(top_n[k], reverse=True)[:n]

        t_preds = [True if p in top_n[wsid] else False for wsid, p in zip(t_wsids, t_probs)]

        # filter based on the boolean list t_preds
        return list(compress(tiles, t_preds))

    @staticmethod
    def batchify(samples):
        # image, label, wsid, x, y
        images = [sample.image for sample in samples if sample]
        images = torch.stack(images, dim=0)
        labels = [sample.label for sample in samples if sample]
        img_paths = [sample.wsid for sample in samples if sample]
        return images, labels, img_paths

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
                print(len(res[0]), len(res[1]), len(res[2]))
                prv = nxt

        return res

    def __label_by_id(self, wsid):

        for row in self.__labels_csv:
            if row['SAMPLE_ID'] in wsid:
                return wsid, row[self.class_type]

        return None, None

    def __read_tiles(self, name, wspaths):
        wsi_dict = defaultdict(list)
        pbar = tqdm(wspaths, desc="Reading Image Folders for {0}".format(name))
        for wspath in pbar:
            metadata = []
            wsid = os.path.splitext(os.path.basename(wspath))[0]
            pbar.set_description("Reading Image Folders for {0}. File: {1}".format(name, wsid))
            filepaths = glob.glob(os.path.join(wspath, "slide", "*", "*.png"))
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

                _, label = self.__label_by_id(wsid)
                label = self.label2id[label]
                metadata.append((filepath, label, wsid, x, y))

            wsi_dict[wspath] = metadata

        return wsi_dict

    def __bin_paths(self, wspaths):
        d = defaultdict(list)
        for wspath in wspaths:
            wsid = os.path.splitext(os.path.basename(wspath))[0]
            _, label = self.__label_by_id(wsid)
            if label:
                label = self.label2id[label]
                d[label].append(wspath)

        return d

    @staticmethod
    def __print_dataset(datasets, names, labels):
        for dataset, name in zip(datasets, names):
            print(name)
            for label in labels:
                print("{0} : {1}".format(label, sum([item[1] == label for k, item in dataset.items()])))

    def read_tiles_and_split(self, image_dir, split):
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

        print(len(wspaths))
        wspaths_train, wspaths_dev, wspaths_test = self.__split_dataset(split, binned_wspaths)

        print(len(wspaths_train), len(wspaths_dev), len(wspaths_test))

        t_train = self.__read_tiles("train", wspaths_train)
        t_dev = self.__read_tiles("dev", wspaths_dev)
        t_test = self.__read_tiles("test", wspaths_test)

        # t_train ... is a dict with wsipath as key and list of tile metadata as value
        return t_train, t_dev, t_test

    def __all_labels(self, name):
        return [row[name] for row in self.__labels_csv]
