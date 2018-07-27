from argparse import ArgumentParser

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from dataset.bic_dataset import BIC_Dataset
from utils import abs_path


def online_stats(X, prev_mean, prev_var, n_seen):
    """
    Converted from John D. Cook
    http://www.johndcook.com/blog/standard_deviation/
    """
    X = X.flatten()
    for i in range(X.shape[0]):

        n_seen += 1

        if prev_mean == 0:
            prev_mean = X[i]
            prev_var = 0.
        else:
            curr_mean = prev_mean + (X[i] - prev_mean) / n_seen
            curr_var = (prev_var * (n_seen - 1) + (X[i] - prev_mean) * (X[i] - curr_mean)) / n_seen
            prev_mean = curr_mean
            prev_var = curr_var

    # n - 1 for sample variance, but numpy default is n
    return prev_mean, prev_var, n_seen


def test_online_stat():
    from numpy.testing import assert_almost_equal
    X = np.random.rand(10000, 50)
    tm = X.mean(axis=0)
    ts = X.std(axis=0)
    sm, ss = online_stats(X)
    assert_almost_equal(tm, sm)
    assert_almost_equal(ts, ss)


def img_norm_calc(dataset):
    data = DataLoader(dataset, batch_size=1, num_workers=8, collate_fn=batchify)
    mu = [0, 0, 0]
    var = [0, 0, 0]
    n_seen = [0, 0, 0]
    t = tqdm(data)
    for image, label, _, _ in t:
        r, c, ch = image[0].shape

        for i in range(ch):
            mu[i], var[i], n_seen[i] = online_stats(image[0][:, :, i], mu[i], var[i], n_seen[i])

        t.desc = "mu={0}, std={1}".format(mu, np.sqrt(var))
    for i in range(3):
        var[i] = np.sqrt(var[i])
    print(mu, var)
    return mu, var


def transform(img):
    img = img.astype(np.float64)
    return img


def calc_mean_var(cfg):
    dataset_manager = BIC_Dataset(train_dir=abs_path(cfg['TRAIN_IMG_DIR']),
                                  test_dir=abs_path(cfg['TEST_IMG_DIR']),
                                  train_save_dir=abs_path(cfg['TRAIN_TILE_DIR']),
                                  test_save_dir=abs_path(cfg['TEST_TILE_DIR']),
                                  label_2_id=cfg['PURE_LABELS'],
                                  tile_size=cfg['TILE_SIZE'],
                                  stride=cfg['STRIDE'],
                                  dev_per=cfg['DEV_PER'],
                                  order=cfg['ORDER'],
                                  randomize=cfg['RANDOMIZE_TRAIN'],
                                  transform=transform,
                                  redo_preprocessing=cfg['REDO_PREPROCESSING'])

    mu, std = img_norm_calc(dataset_manager.train)

    with open("mean_std.txt", 'w') as f:
        f.write("mu=\n")
        f.write(mu)
        f.write("std=\n")
        f.write(std)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-y', '--yaml', dest="yaml", help="Yaml config file")

    args = parser.parse_args()
    with open(args.yaml, 'r') as cfgfile:
        config = yaml.load(cfgfile)

    calc_mean_var(config)
