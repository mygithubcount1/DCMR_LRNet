import os
import math
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from utils import train_one_epoch_acc, evaluate_acc
from model import MyModel

from dataset import FGDataset as BaseFGDataset
from dataset_soycultivar200 import FGDataset as SoyCultivar200FGDataset
import torchvision.transforms.functional as TF

import sys
import logging

script_name = os.path.basename(sys.argv[0])
log_filename = os.path.splitext(script_name)[0] + ".log"
log_filename = os.path.join("./logs", log_filename)

if os.path.exists(f"./logs") is False:
    os.makedirs(f"./logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)


class TopCrop:
    def __init__(self, crop_size, crop_dim_vertical):
        self.crop_size = crop_size
        self.crop_dim_vertical = crop_dim_vertical

    def __call__(self, img):
        width, height = img.size

        if self.crop_dim_vertical:
            left = (width - self.crop_size) // 2
            return TF.crop(img, 0, left, self.crop_size, self.crop_size)
        else:
            top = (height - self.crop_size) // 2
            return TF.crop(img, top, 0, self.crop_size, self.crop_size)


class BottomCrop:
    def __init__(self, crop_size, crop_dim_vertical):
        self.crop_size = crop_size
        self.crop_dim_vertical = crop_dim_vertical

    def __call__(self, img):
        width, height = img.size
        if self.crop_dim_vertical:
            top = height - self.crop_size
            left = (width - self.crop_size) // 2
            return TF.crop(img, top, left, self.crop_size, self.crop_size)
        else:
            top = (height - self.crop_size) // 2
            left = width - self.crop_size
            return TF.crop(img, top, left, self.crop_size, self.crop_size)


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rawdata_root = ""
    anno_train = ""
    anno_test = ""
    crop_dim_vertical = True
    dataset_class = BaseFGDataset
    center_resize_size=args.resize_size
    resize_size = args.resize_size
    crop_size = args.crop_size
    args.save_dirname = f'{args.save_dirname}_{args.dataset}_seed{args.seed}'

    if args.dataset == 'COTTON':
        args.num_classes = 80
        crop_dim_vertical = False
        center_resize_size=512
        rawdata_root = './dataset/COTTON/images_pytorch/'
        anno_train = './dataset/COTTON/anno/train.txt'
        anno_test = './dataset/COTTON/anno/test.txt'
    elif args.dataset == "SoyAgeing":
        args.num_classes = 198
        args.save_dirname = f'{args.save_dirname}_stage{args.stage}'
        rawdata_root = f'./dataset/SoyAgeing/{args.stage}/images_pytorch/'
        anno_train = f'./dataset/SoyAgeing/{args.stage}/anno/train.txt'
        anno_test = f'./dataset/SoyAgeing/{args.stage}/anno/test.txt'
    elif args.dataset == 'SoyCultivar200':
        args.num_classes = 200
        rawdata_root = r'./dataset/SoyCultivar200_dataset'
        if args.swap:
            args.save_dirname = f'{args.save_dirname}_position{args.position}_swap'
            anno_train = rf'./dataset/200_anno2/train_swap_{args.position}.txt'
            anno_test = rf'./dataset/200_anno2/test_swap_{args.position}.txt'
        else:
            args.save_dirname = f'{args.save_dirname}_position{args.position}'
            anno_train = rf'./dataset/200_anno2/train_{args.position}.txt'
            anno_test = rf'./dataset/200_anno2/test_{args.position}.txt'
        dataset_class = SoyCultivar200FGDataset
        resize_size = 512
        center_resize_size =512
    elif args.dataset == "soybean200":
        args.num_classes = 200
        rawdata_root = './dataset/soybean200/images_pytorch/'
        anno_train = './dataset/soybean200/anno/train.txt'
        anno_test = './dataset/soybean200/anno/test.txt'
    elif args.dataset == "SoyGene":
        args.num_classes = 1110
        rawdata_root = './dataset/SoyGene/images_pytorch'
        anno_train = './SoyGene/anno/train.txt'
        anno_test = './dataset/SoyGene/anno/test.txt'
    elif args.dataset == "SoyGlobal":
        args.num_classes = 1938
        rawdata_root = './dataset/SoyGlobal/images_pytorch'
        anno_train = './dataset/SoyGlobal/anno/train.txt'
        anno_test = './dataset/SoyGlobal/anno/test.txt'
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        "val_center": transforms.Compose([
            transforms.Resize(center_resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]),
        "val_top": transforms.Compose([
            transforms.Resize(resize_size),
            TopCrop(crop_size, crop_dim_vertical),
            transforms.ToTensor(),
        ]),
        "val_bottom": transforms.Compose([
            transforms.Resize(resize_size),
            BottomCrop(crop_size, crop_dim_vertical),
            transforms.ToTensor(),
        ])
    }
    logging.info(args)
    logging.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    nw = args.nw
    logger.info('Using {} dataloader workers every process'.format(nw))

    train_dataset = dataset_class(rawdata_root, anno_train, data_transform, is_train=True)
    val_dataset = dataset_class(rawdata_root, anno_test, data_transform, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=nw)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    model = MyModel(args).to(device)

    checkpoint = torch.load(args.trained_weights, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)


    # validate
    val_loss, val_acc = evaluate_acc(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=1)

    logger.info(
        f"[epoch {1}]  val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")


if __name__ == '__main__':
    def set_seed(seed):
        # seed init.
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # torch seed init.
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False  # train speed is slower after enabling this opts.

        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

        # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
        torch.use_deterministic_algorithms(True)


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--nw', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--model_name', type=str, default='efficientnet_b0')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--save_dirname', type=str, default='weights')
    parser.add_argument('--weights_path', type=str, default='./weights/efficientnet_b0_ra-3dd342df.pth')

    parser.add_argument('--trained_weights', type=str, default='./weights_COTTON_seed3407/model-max_acc.pth')

    parser.add_argument('--save_model', type=bool, default=False)

    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--stage', type=str, default='R6')
    # soybean200、SoyAgeing、COTTON
    parser.add_argument('--dataset', type=str, default='COTTON')

    parser.add_argument('--position', type=str, default='U')

    # Whether to use swap cross-validation for the SoyCultivar200 dataset
    parser.add_argument('--swap', type=bool, default=False)

    parser.add_argument('--resize_size', type=int, default=590)
    parser.add_argument('--crop_size', type=int, default=448)

    opt = parser.parse_args()

    main(opt)