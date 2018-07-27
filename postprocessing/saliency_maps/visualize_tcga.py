import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, transforms

from dataset.tcga_dataset import TCGADataset
from models.alexnet import alexnet
from postprocessing.saliency_maps.cam import CAM
from utils import abs_path


def visualize_tcga(cfg):
    inp_dir = cfg['INPUT_DIR']
    res_dir = cfg['RES_DIR']
    meta_name = cfg['META_NAME']

    transform = Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    model_save_file = cfg['MODEL_SAVE_FILE']
    classes = {0: "mitosis", 1: "no-mitosis"}

    db_man = TCGADataset(class_type=cfg['CLASS_TYPE'],
                         image_dir=cfg['IMG_DIR'],
                         label_filepath=cfg['LABEL_FILE'],
                         split=cfg['SPLIT'],
                         label2id=cfg['PURE_LABELS'],
                         transform=transform,
                         filter_model=None,
                         filter_percent=cfg['MITOSES_FILTER_PERCENT'])

    classes = {v: k for k, v in db_man.label2id.items()}

    data_loader = DataLoader(dataset=db_man.train,
                             batch_size=1,
                             num_workers=1,
                             collate_fn=db_man.batchify)

    model = alexnet(pretrained=True, num_classes=db_man.no_labels)
    model = model.cuda()
    checkpoint = torch.load(model_save_file)
    model.load_state_dict(checkpoint['state_dict'])

    cam = CAM()

    cam.visualize(data_loader, model, classes, cfg['VIS_DIR'])

