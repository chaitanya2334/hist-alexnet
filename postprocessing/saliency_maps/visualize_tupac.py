import gc

import cv2
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, transforms
from tqdm import tqdm

from dataset.tupac_dataset import TUPACDataset
from models.alexnet import alexnet
from postprocessing.saliency_maps.cam import CAM
from utils import abs_path, argmax
import numpy as np
from pathlib import Path


def save_image(save_dir, file_name, image):
    dir1 = os.path.join(*Path(file_name).parts[-3:])
    save_path = os.path.join(save_dir, dir1)
    cv2.imwrite(save_path, image)


def visualize_tupac(cfg):
    inp_dir = cfg['INPUT_DIR']
    res_dir = cfg['RES_DIR']
    meta_name = cfg['META_NAME']

    transform = Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    model_save_file = abs_path(res_dir, "mitoses", cfg['MODEL_SAVE_FILE'])
    classes = {0: "mitosis", 1: "no-mitosis"}

    db_man = TUPACDataset(wsi_dir=abs_path(inp_dir, cfg['WSI_DIR']),
                          tiles_dir=abs_path(inp_dir, meta_name, cfg['IMG_DIR']),
                          label_dir=abs_path(inp_dir, meta_name, cfg['LABEL_DIR']),
                          label_2_id=cfg['PURE_LABELS'],
                          tile_size=cfg['TILE_SIZE'],
                          split=cfg['SPLIT'],
                          mu=cfg['MEAN'],
                          std=cfg['STD'],
                          randomize=cfg['RANDOMIZE_TRAIN'],
                          transform=transform,
                          pre_visualize=cfg['PRE_VISUALIZE'],
                          vis_dir=abs_path(inp_dir, cfg['VIS_DIR']))

    data_loader = DataLoader(dataset=db_man.train,
                             batch_size=1,
                             num_workers=1,
                             collate_fn=db_man.batchify)
    cam = CAM()

    model = alexnet(pretrained=True, num_classes=2)
    model = model.cuda()
    checkpoint = torch.load(model_save_file)
    model.load_state_dict(checkpoint['state_dict'])

    for image, label, image_path in tqdm(data_loader, desc="visualizing", total=np.math.ceil(len(data_loader))):

        label = label[0]
        image_path = image_path[0]
        seq_out = model(Variable(image).cuda())

        pred = argmax(seq_out)
        if label == pred[0]:
            res = cam.gen_cam_saliency(image, model, 'features', classes)

            if classes[label] == "mitosis":
                _dir = "input/SM/mitoses"
            else:
                _dir = "input/SM/non_mitoses"

            for i, (orig, result, heatmap) in enumerate(res):
                # save image
                save_dir = os.path.join(_dir, "saliency_maps")
                os.makedirs(save_dir, exist_ok=True)

                save_image(save_dir, Path(image_path).stem + "_" + classes[i] + "_original.png", orig)
                save_image(save_dir, Path(image_path).stem + "_" + classes[i] + ".png", result)
                save_image(save_dir, Path(image_path).stem + "_" + classes[i] + "_heatmap.png", heatmap)

            del res
        gc.collect()