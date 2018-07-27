import shutil
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

from dataset.tcga_dataset import TCGADataset
from model.alexnet import alexnet
from torchvision.transforms import Compose
from tqdm import tqdm
from model.utils import to_scalar
from postprocessing.evaluator import Evaluator
from samplers.fast_wr_sampler import FastWeightedRandomSampler
from utils import abs_path, argmax
import numpy as np
import preprocessing.normal_staining as ns
import os


def train_a_epoch(name, data, model, optimizer, criterion, res_file, lr, batch_size, label2id, en):
    model = model.train(True)

    evaluator = Evaluator(name, label2id)
    print("evaluator loaded")
    i = 0
    pbar = tqdm(data, desc=name, total=np.math.ceil(len(data)))
    for images, labels, _ in pbar:
        # zero the parameter gradients
        sys.stdout.flush()
        optimizer.zero_grad()
        if i == 0:
            pbar.total = np.math.ceil(len(data))
        inp = Variable(images.cuda())
        seq_out = model(inp)
        pred = argmax(seq_out)

        loss = criterion(seq_out, Variable(torch.cuda.LongTensor(labels)))
        pbar.set_description("{0} loss: {1}".format(name, to_scalar(loss)))
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
    model = model.eval()

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


def resume(resume_file, model, optimizer):
    best_res_val = 0.0
    best_epoch = 0
    start_epoch = 0

    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint.get('epoch', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        best_res_val = checkpoint.get('best_f1', 0)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint.get('epoch', 0)))
        except (KeyError, RuntimeError):
            print("Unable to load checkpoint {} (epoch {}) correctly. "
                  "Starting from scratch".format(resume_file, checkpoint.get('epoch', None)))
            best_res_val = 0.0
            best_epoch = 0
            start_epoch = 0

    else:
        print("=> no checkpoint found at '{}'".format(resume_file))

    return model, optimizer, start_epoch, best_epoch, best_res_val


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
                balance_train=False,
                max_epochs=100,
                resume_file="checkpoint.dict"):
    # init model
    model = alexnet(pretrained=pretrain, num_classes=no_labels)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    print("Model Loaded")

    # Turn on cuda
    model = model.cuda()
    print("Model loaded in cuda memory")

    optimizer.zero_grad()
    model.zero_grad()

    # init loss criteria
    criterion = torch.nn.CrossEntropyLoss()

    model, optimizer, start_epoch, best_epoch, best_res_val = resume(resume_file, model, optimizer)

    for epoch in range(start_epoch, max_epochs):
        print('-' * 40)
        print("EPOCH = {0}".format(epoch))
        print('-' * 40)
        if balance_train:
            class_weights, sample_labels = dataset.train.get_sample_weights(uniform=True)
            sampler = FastWeightedRandomSampler(class_weights=class_weights,
                                                sample_labels=sample_labels,
                                                max_nsamples=len(sample_labels),
                                                replacement=False,
                                                get_all=False)
            print("Using weighted random sampler. n_classes = {0}".format(len(class_weights)))
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(dataset=dataset.train,
                                  batch_size=batch_size,
                                  num_workers=no_workers,
                                  shuffle=shuffle,
                                  sampler=sampler,
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
            'best_epoch': best_epoch,
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

    print("Prepping Dataset ...")
    transform = Compose([
        ns.normalize_staining,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if cfg['MITOSES_MODEL_FILE']:
        filter_model = cfg['MITOSES_MODEL_FILE']
    else:
        filter_model = None

    db_man = TCGADataset(class_type=cfg['CLASS_TYPE'],
                         image_dir=cfg['IMG_DIR'],
                         label_filepath=cfg['LABEL_FILE'],
                         split=cfg['SPLIT'],
                         label2id=cfg['PURE_LABELS'],
                         transform=transform,
                         filter_model=filter_model,
                         filter_percent=cfg['MITOSES_FILTER_PERCENT'])

    test_loader = DataLoader(dataset=db_man.test,
                             batch_size=cfg['BATCH_SIZE'],
                             num_workers=cfg['NUM_WORKERS'],
                             collate_fn=db_man.batchify)

    the_model, best_en = build_model(db_man,
                                     label2id=db_man.label2id,
                                     no_labels=db_man.no_labels,
                                     batch_size=cfg['BATCH_SIZE'],
                                     no_workers=cfg['NUM_WORKERS'],
                                     train_res_file=cfg['TRAIN_RES_FILE'],
                                     test_res_file=cfg['TEST_RES_FILE'],
                                     dev_res_file=cfg['DEV_RES_FILE'],
                                     model_save_file=cfg['MODEL_SAVE_FILE'],
                                     train_till_epoch=cfg["TRAIN_TILL_EPOCH"],
                                     pretrain=cfg['PRETRAIN'],
                                     lr=cfg['LEARNING_RATE'],
                                     weight_decay=cfg['L2_REG'],
                                     balance_train=cfg['BALANCE_TRAIN'],
                                     max_epochs=cfg['MAX_EPOCH'],
                                     resume_file=cfg['CHECKPOINT_FILE'])

    print("running test")
    test_eval, img_paths = test_a_epoch(name="test",
                                        data=test_loader,
                                        model=the_model,
                                        result_file=cfg['TEST_RES_FILE'],
                                        label2id=db_man.label2id,
                                        lr=cfg['LEARNING_RATE'],
                                        batch_size=cfg['BATCH_SIZE'],
                                        en=best_en)

    test_eval.print_results()
