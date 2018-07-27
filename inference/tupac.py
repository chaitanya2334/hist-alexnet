import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.tupac_dataset import TUPACDataset
from model.alexnet import alexnet
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.mitosis_dataset import MitosisDataset
from model.inception3 import inception_v3, InceptionAux
from model.utils import to_scalar
from postprocessing.evaluator import Evaluator
from utils import abs_path, argmax
import numpy as np
import os


def test_a_epoch(name, data, model, result_file, label2id):
    model.training = False

    pred_list = []
    true_list = []
    evaluator = Evaluator(name, label2id)

    for images, labels in tqdm(data, desc=name, total=np.math.ceil(len(data))):
        inp = Variable(images.cuda())

        seq_out = model(inp)

        pred = argmax(seq_out)
        evaluator.append_data(0, pred, labels)

        # print(predicted)
        pred_list.extend(pred)
        true_list.extend(labels)

    evaluator.gen_results()
    evaluator.print_results()
    evaluator.write_results(result_file, '')

    return evaluator, pred_list, true_list


def single_run(cfg):
    inp_dir = cfg['INPUT_DIR']
    res_dir = cfg['RES_DIR']
    meta_name = cfg['META_NAME']

    print("Prepping Dataset ...")

    transform = Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    db_man = TUPACDataset(image_dir=abs_path(inp_dir, meta_name, cfg['IMG_DIR']),
                          label_dir=abs_path(inp_dir, meta_name, cfg['LABEL_DIR']),
                          label_2_id=cfg['PURE_LABELS'],
                          tile_size=cfg['TILE_SIZE'],
                          split=cfg['SPLIT'],
                          mu=cfg['MEAN'],
                          std=cfg['STD'],
                          randomize=cfg['RANDOMIZE_TRAIN'],
                          transform=transform,
                          pre_visualize=cfg['PRE_VISUALIZE'])

    test_loader = DataLoader(dataset=db_man.full,
                             batch_size=cfg['BATCH_SIZE'],
                             num_workers=cfg['NUM_WORKERS'],
                             collate_fn=db_man.batchify)

    print("Prepping Model ...")
    the_model = torch.load(cfg['MODEL_SAVE_FILE'])

    print("running test")
    test_eval, pred_list, true_list = test_a_epoch(name="test",
                                                   data=test_loader,
                                                   model=the_model,
                                                   result_file=abs_path(res_dir, meta_name, cfg['TEST_RES_FILE']),
                                                   label2id=cfg['PURE_LABELS'])

    test_eval.print_results()

