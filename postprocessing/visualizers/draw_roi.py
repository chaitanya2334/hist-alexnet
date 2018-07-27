import glob

import cv2
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

from tqdm import tqdm

Tile = namedtuple('Tile', ['true', 'pred', 'x', 'y', 'parent'])


class TileVisualizer:
    def __init__(self, wsi_dir, save_dir, pos_label, neg_label, tile_size):
        self.wsi_dir = wsi_dir
        self.vis_dir = save_dir
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.tile_size = tile_size
        self.tiles = None

    @staticmethod
    def _bin_by_img_id(tiles):
        # pred is a list of namedtuple: [(label, x, y, parent)]
        pos_tiles = {}
        neg_tiles = {}
        for entry in tiles:
            if entry.parent not in pos_tiles:
                pos_tiles[entry.parent] = []

            pos_tiles[entry.parent].append(entry)

        return pos_tiles

    @staticmethod
    def draw(img, bb, color):
        # draw rectangles on the original image
        x, y, w, h = tuple(map(int, bb))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        return img

    def visualize(self, trues, preds, tile_paths, show_pos=True, show_neg=False):
        tiles = self._make_tiles(trues, preds, tile_paths)
        binned_tiles = self._bin_by_img_id(tiles)

        for img_path, tiles in tqdm(binned_tiles.items(), desc="drawing"):
            img = cv2.imread(os.path.join(self.wsi_dir, img_path))
            for tile in tiles:
                bb = (tile.x, tile.y, self.tile_size, self.tile_size)

                if show_pos and tile.true == self.pos_label:
                    img = self.draw(img, bb, color=(0, 255, 0))
                if show_neg and tile.true == self.neg_label:
                    img = self.draw(img, bb, color=(255, 255, 255))

            save_path = os.path.join(self.vis_dir, os.path.splitext(img_path)[0] + ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

    def vis_pred(self, preds, trues, tile_paths, tp_color=None, tn_color=None, fp_color=None, fn_color=None):
        tiles = self._make_tiles(trues, preds, tile_paths)
        binned_tiles = self._bin_by_img_id(tiles)

        for img_path, tiles in tqdm(binned_tiles.items(), desc="drawing"):
            img = cv2.imread(os.path.join(self.wsi_dir, img_path))
            for tile in tiles:
                bb = (tile.x, tile.y, self.tile_size, self.tile_size)

                if tp_color and tile.true == tile.pred == self.pos_label:
                    img = self.draw(img, bb, color=tp_color)
                if tn_color and tile.true == tile.pred == self.neg_label:
                    img = self.draw(img, bb, color=tn_color)
                if fp_color and tile.true != tile.pred and tile.true == self.pos_label:
                    img = self.draw(img, bb, color=fp_color)
                if fn_color and tile.true != tile.pred and tile.true == self.neg_label:
                    img = self.draw(img, bb, color=fn_color)

            save_path = os.path.join(self.vis_dir, os.path.splitext(img_path)[0] + ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

    @staticmethod
    def _make_tiles(trues, preds, tile_paths):
        ret = []
        for true, pred, tile_path in tqdm(zip(trues, preds, tile_paths), total=len(trues), desc="Making tiles for visualization"):
            # Step. 1 get x, y
            # get basename
            tile_name = os.path.basename(tile_path)
            # remove ext
            tile_name = os.path.splitext(tile_name)[0]
            # get x, y pos
            x, y, r = tile_name.split("-")

            # Step. 2 get img_id which is a combination of its two parent dirs
            # get the original wsi name
            wsi_dir = os.path.dirname(tile_path)
            wsi_name = os.path.basename(wsi_dir) + ".tif"

            # get the parent dir's basename of the wsi name
            wsi_dir_name = os.path.basename(os.path.dirname(wsi_dir))
            parent = os.path.join(wsi_dir_name, wsi_name)

            ret.append(Tile(true=true, pred=pred, x=x, y=y, parent=parent))
        return ret


if __name__ == '__main__':
    image_dir = "input/tupac_tiled/"
