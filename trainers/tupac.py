import shutil
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

from dataset.tupac_dataset import TUPACDataset
from model.alexnet_mitoses import alexnet
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.mitosis_dataset import MitosisDataset
from model.inception3 import inception_v3, InceptionAux
from model.utils import to_scalar
from postprocessing.evaluator import Evaluator
from postprocessing.visualizers.draw_roi import TileVisualizer
from preprocessing.normal_staining import normalize_staining
from utils import abs_path, argmax
import numpy as np
import os


def train_a_epoch(name, data, model, optimizer, criterion, res_file, lr, batch_size, label2id, en):
    model.training = True

    evaluator = Evaluator(name, label2id)
    print("evaluator loaded")
    i = 0
    for images, labels, _ in tqdm(data, desc=name, total=np.math.ceil(len(data))):
        # zero the parameter gradients
        sys.stdout.flush()
        optimizer.zero_grad()
        model.zero_grad()
        # image = crop(sample.image, cfg.MAX_IMG_SIZE)

        inp = Variable(images.cuda())
        # inp = torch.transpose(inp, 1, 3)
        seq_out = model(inp)

        pred = argmax(seq_out)
        loss = criterion(seq_out, Variable(torch.cuda.LongTensor(labels)))

        evaluator.append_data(to_scalar(loss), pred, labels)
        loss.backward()
        optimizer.step()
        i += 1

    print("Training Done")
    evaluator.gen_results()
    evaluator.print_results()
    evaluator.write_results(res_file,
                            "epoch = {2}; GOOGLE NET; lr={0}; batch_size={1}".format(lr,
                                                                                     batch_size,
                                                                                     en))

    return model


def test_a_epoch(name, data, model, result_file, label2id, lr, batch_size, en):
    model.training = False

    full_img_paths = []
    evaluator = Evaluator(name, label2id)

    for images, labels, img_paths in tqdm(data, desc=name, total=np.math.ceil(len(data))):
        inp = Variable(images.cuda())

        seq_out = model(inp)

        pred = argmax(seq_out)
        evaluator.append_data(0, pred, labels)

        # print(predicted)
        full_img_paths.extend(img_paths)

    evaluator.gen_results()
    evaluator.print_results()
    evaluator.write_results(result_file,
                            "epoch = {2}; GOOGLE NET; lr={0}; batch_size={1}".format(lr,
                                                                                     batch_size,
                                                                                     en))

    return evaluator, full_img_paths


def build_model(dataset,
                label2id,
                no_labels=2,
                batch_size=16,
                no_workers=8,
                train_res_file="train_results.txt",
                test_res_file="test_results.txt",
                dev_res_file="dev_results.txt",
                model_save_file="save.m",
                train_till_epoch=20,
                pretrain=True,
                lr=0.003,
                weight_decay=0.002,
                max_epochs=100,
                resume_file="checkpoint.dict"):
    # init model
    model = alexnet(pretrained=pretrain, num_classes=no_labels)
    optimizer = torch.optim.Adadelta(model.parameters(), lr, weight_decay)
    print("Model Loaded")

    # Turn on cuda
    model = model.cuda()
    print("Model loaded in cuda memory")

    optimizer.zero_grad()
    model.zero_grad()

    # init loss criteria
    criterion = torch.nn.CrossEntropyLoss()

    best_res_val = 0.0
    best_epoch = 0
    start_epoch = 0

    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            best_res_val = checkpoint['best_f1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    for epoch in range(start_epoch, max_epochs):
        print('-' * 40)
        print("EPOCH = {0}".format(epoch))
        print('-' * 40)

        train_loader = DataLoader(dataset=dataset.train,
                                  batch_size=batch_size,
                                  num_workers=no_workers,
                                  shuffle=True,
                                  collate_fn=dataset.batchify)

        print("train_loader ready")
        model = train_a_epoch(name="train",
                              data=train_loader,
                              model=model,
                              optimizer=optimizer,
                              criterion=criterion,
                              res_file=abs_path(train_res_file),
                              lr=lr,
                              batch_size=batch_size,
                              label2id=label2id,
                              en=epoch)

        dev_loader = DataLoader(dataset=dataset.dev,
                                batch_size=batch_size,
                                num_workers=no_workers,
                                shuffle=True,
                                collate_fn=dataset.batchify)

        dev_eval, _ = test_a_epoch(name="dev",
                                   data=dev_loader,
                                   model=model,
                                   result_file=abs_path(dev_res_file),
                                   label2id=label2id,
                                   lr=lr,
                                   batch_size=batch_size,
                                   en=epoch)

        dev_eval.verify_results()

        is_best = dev_eval.f > best_res_val
        save_checkpoint({
            'epoch': epoch + 1,
            'model': "alexnet",
            'state_dict': model.state_dict(),
            'best_f1': best_res_val,
            'optimizer': optimizer.state_dict(),
        },
            is_best,
            resume_file,
            model_save_file)

        if epoch == 0 or dev_eval.f > best_res_val:
            best_epoch = epoch
            best_res_val = dev_eval.f

        print("current dev score: {0}".format(dev_eval.f))
        print("best dev score: {0}".format(best_res_val))
        print("best_epoch: {0}".format(str(best_epoch)))

        if 0 < train_till_epoch <= (epoch - best_epoch):
            break

    # print("Loading Best Model ...")
    best_model_checkpoint = torch.load(model_save_file)
    model.load_state_dict(best_model_checkpoint['state_dict'])

    return model, best_epoch


def save_checkpoint(state, is_best, cp_file, model_save_file):
    torch.save(state, cp_file)
    if is_best:
        shutil.copyfile(cp_file, model_save_file)


def single_run(cfg):
    inp_dir = cfg['INPUT_DIR']
    res_dir = cfg['RES_DIR']
    meta_name = cfg['META_NAME']

    print("Prepping Dataset ...")

    transform = Compose([
        normalize_staining,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

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

    test_loader = DataLoader(dataset=db_man.test,
                             batch_size=cfg['BATCH_SIZE'],
                             num_workers=cfg['NUM_WORKERS'],
                             collate_fn=db_man.batchify)

    the_model, best_en = build_model(db_man,
                                     label2id=cfg['PURE_LABELS'],
                                     no_labels=cfg['NUM_LABELS'],
                                     batch_size=cfg['BATCH_SIZE'],
                                     no_workers=cfg['NUM_WORKERS'],
                                     train_res_file=abs_path(res_dir, meta_name, cfg['TRAIN_RES_FILE']),
                                     test_res_file=abs_path(res_dir, meta_name, cfg['TEST_RES_FILE']),
                                     dev_res_file=abs_path(res_dir, meta_name, cfg['DEV_RES_FILE']),
                                     model_save_file=abs_path(res_dir, meta_name, cfg['MODEL_SAVE_FILE']),
                                     train_till_epoch=cfg["TRAIN_TILL_EPOCH"],
                                     pretrain=cfg['PRETRAIN'],
                                     lr=cfg['LEARNING_RATE'],
                                     weight_decay=cfg['L2_REG'],
                                     max_epochs=cfg['MAX_EPOCH'],
                                     resume_file=abs_path(res_dir, meta_name, cfg['CHECKPOINT_FILE']))

    print("running test")

    test_eval, img_paths = test_a_epoch(name="test",
                                        data=test_loader,
                                        model=the_model,
                                        result_file=abs_path(res_dir, meta_name, cfg['TEST_RES_FILE']),
                                        label2id=cfg['PURE_LABELS'],
                                        lr=cfg['LEARNING_RATE'],
                                        batch_size=cfg['BATCH_SIZE'],
                                        en=best_en)

    test_eval.print_results()

    vis = TileVisualizer(wsi_dir=abs_path(inp_dir, cfg['WSI_DIR']),
                         save_dir=abs_path(inp_dir, cfg['TEST_VIS_DIR']),
                         pos_label=cfg['PURE_LABELS']["mitosis"],
                         neg_label=cfg['PURE_LABELS']["no-mitosis"],
                         tile_size=cfg['TILE_SIZE'])

    vis.vis_pred(preds=test_eval.pred,
                 trues=test_eval.true,
                 tile_paths=img_paths,
                 tp_color=(255, 0, 0),
                 fp_color=(0, 255, 0),
                 fn_color=(0, 0, 255))
