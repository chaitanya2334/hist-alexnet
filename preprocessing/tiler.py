import gc
import glob
import os
import random
from time import sleep

import cv2
from tqdm import tqdm

import csv_utils
import numpy as np

from preprocessing.normal_staining import normalize_staining, is_norm_stainable


class Tiler:
    def __init__(self, size, stride, padding, label_dir, src_dir, dest_dir, pos_label, neg_label):
        self.tile_size = size
        self.padding = padding
        self.stride = stride
        self.label_dir = label_dir
        self.srcdir = src_dir
        self.destdir = dest_dir
        self.pos_label = pos_label
        self.neg_label = neg_label

    def get_tile(self, wsi, i, j):
        tile = wsi[i:i + self.tile_size, j:j + self.tile_size, :]
        name = str(i) + '-' + str(j)
        return tile, name

    def tile(self, image, path, pos=False, neg=0):
        pad = int(self.padding)
        image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'minimum')
        row, col, ch = image.shape

        if pos:
            t = int(((row - self.tile_size + 1)/ self.stride) * ((col - self.tile_size + 1) / self.stride))
            with tqdm(total=t, desc="pos tiles") as pbar:
                for i in range(0, row - self.tile_size + 1, self.stride):
                    for j in range(0, col - self.tile_size + 1, self.stride):
                        pbar.update()
                        lbl = self.get_label(path, i, j)
                        if lbl == self.pos_label:
                            yield self.get_tile(image, i, j)

        if neg != 0:
            c = 0
            while c < neg:
                i = random.choice(range(0, row - self.tile_size + 1, self.stride))
                j = random.choice(range(0, col - self.tile_size + 1, self.stride))
                lbl = self.get_label(path, i, j)
                if lbl == self.neg_label:
                    c += 1
                    yield self.get_tile(image, i, j)

    def save(self, images, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if isinstance(images[0], tuple):
            for img, img_name in images:
                cv2.imwrite(os.path.join(path, img_name + ".png"), img)

        else:
            for i, img in enumerate(images):
                cv2.imwrite(os.path.join(path, str(i) + ".png"), img)

    def find_all_images(self, path, img_ext):
        files = glob.iglob(path + '/**/*' + img_ext, recursive=True)
        cpt = sum([len(files) for r, d, files in os.walk(path)])
        for file in tqdm(files, total=cpt):
            img = cv2.imread(file)
            img = img.astype(np.float64)

            yield img, file

    def refactor_filestruct(self, srcpath, labelfile, destpath):
        labels = ['Benign', 'In Situ', 'Invasive', 'Normal']
        i = 0
        rows = []
        for label in labels:
            path = os.path.join(srcpath, label)
            files = glob.glob(path + '/*.tif')

            for file in tqdm(files):
                img = cv2.imread(file)
                cv2.imwrite(os.path.join(destpath, "{0}.tif".format(i)), img)

                rows.append({'id': i, 'label': label})
                i += 1

        csv_utils.write(labelfile, rows, headers=['id', 'label'], append=True)

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def rotate(self, img, name, angles=(90, 180, 270, 360)):
        rot_img_name_pairs = [(self.rotate_bound(img, angle), name + "-" + str(angle)) for angle in angles]

        return zip(*rot_img_name_pairs)

    def vert_reflect(self, imgs):
        ref_imgs = []
        for img in imgs:
            ref_imgs.extend([img, cv2.flip(img, 1)])

        return ref_imgs

    def get_label(self, path, x, y):
        # assuming path = <root>/<img_id>/<img_name>.png
        # path = <root>/tiles/12/04/0-600-90.png
        img_id = os.path.splitext(path)[0].replace(self.srcdir + '/', '')
        # img_id = 12/04

        # x = 0, y = 600, r = 90
        lbl_path = os.path.join(self.label_dir, img_id + ".csv")
        # lbl_path = <root>/labels/12/04.csv
        true_labels = []
        if os.path.isfile(lbl_path):
            true_labels = csv_utils.read(lbl_path, is_headers=False)

        # TODO verify row columns for label file
        labels = [(int(row[0]), int(row[1])) for row in true_labels]
        # labels = [(70, 1782)]
        for label in labels:
            if (int(x) < int(label[0]) <= int(x) + self.tile_size) and \
                    (int(y) < int(label[1]) <= int(y) + self.tile_size):
                return self.pos_label

        return self.neg_label

    def run(self, normalize=True, rot=True):
        def rot_and_save(img, tn):
            if rot:
                images, tile_names = self.rotate(img, name=tn)
            else:
                images = [img]
                tile_names = [tn]

            self.save(list(zip(images, tile_names)), new_dir)
            images = []
            tile_names = []
            gc.collect()

        for wsi, wsi_path in self.find_all_images(self.srcdir, img_ext='.tif'):
            if normalize:
                image = normalize_staining(wsi)
            else:
                image = wsi

            new_dir = os.path.splitext(wsi_path)[0]
            new_dir = new_dir.replace(self.srcdir, self.destdir)
            row, col, ch = image.shape
            # get all the positive tiles first, keep a count
            c = 0
            for image, tile_name in self.tile(image, wsi_path, pos=True):
                c += 1
                if is_norm_stainable(image):
                    rot_and_save(image, tile_name)

            for image, tile_name in tqdm(self.tile(wsi, wsi_path, neg=c), total=c,
                                         desc="tiling neg from img size:({0}x{1})".format(col, row)):
                if is_norm_stainable(image):
                    rot_and_save(image, tile_name)

    def rot_and_reflect(self, srcdir, destdir, subdir):
        path = os.path.join(srcdir, subdir)
        for image, name in self.find_all_images(path, img_ext='.png'):
            rot_images = self.rotate([image])
            ref_images = self.vert_reflect(rot_images)
            self.save(ref_images, os.path.join(destdir, subdir, name))


def main():
    root = "/fs/scratch/osu1522/TUPAC/mitoses"
    srcdir = "train"
    destdir = "train_tiled"
    label_dir = "labels"

    t = Tiler(size=224,
              stride=15,
              padding=0,
              label_dir=os.path.join(root, label_dir),
              src_dir=os.path.join(root, srcdir),
              dest_dir=os.path.join(root, destdir),
              pos_label="mitoses",
              neg_label="no-mitoses")

    t.run(normalize=False, rot=True)


if __name__ == '__main__':
    main()
