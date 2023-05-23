#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# import matplotlib.pyplot as plt
from utils.utils import DiceLoss
from torch.utils.data import DataLoader
from dataset.dataset_Miccai import Task1Dataset
from torch.nn.modules.loss import CrossEntropyLoss
import argparse
from tqdm import tqdm
import os
from torchvision import transforms
from utils.test_Miccai import inference
from model.MTUNet import MTUNet
import numpy as np
from medpy.metric import dc,hd95
from utils.focal_loss import FocalLoss
import random

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=3, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=100)
parser.add_argument("--img_size", default=512)
parser.add_argument("--save_path", default="./checkpoint/Miccai/mtunet")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--root_dir", default="./data/Miccai")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=3)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument("--patches_size", default=32)
parser.add_argument("--n-skip", default=1)
args = parser.parse_args()
args.target = 0

def set_seed(Seed=100):
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)

set_seed()

model=MTUNet(args.num_classes, args.img_size) # 9

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 预处理
train_transform = A.Compose([
        # A.GaussianBlur(p=0.3),
        # A.HorizontalFlip(p=0.3),
        # A.RandomBrightnessContrast(p=0.3),
        # A.ShiftScaleRotate(p=0.3),
        
        A.Flip(), # 随机翻转
        A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=90), # 随机旋转
        A.OneOf([
            A.RandomBrightnessContrast(p=1),    # 随机亮度
            A.RandomGamma(p=1),                 # 随机Gamma
        ]),
        ##A.CoarseDropout(max_height=5, min_height=1, max_width=512, min_width=51, mask_fill_value=0),
        #A.OneOf([
        #    A.Sharpen(p=1),
        #    A.Blur(blur_limit=3, p=1),
        #    A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
        #    A.NoOp(p=1)
        #]),
        #A.RandomResizedCrop(512, 512, p=0.2),
        #A.GridDistortion(p=0.2),
        #A.CoarseDropout(max_height=128, min_height=32, max_width=128, min_width=32, max_holes=3, p=0.2, mask_fill_value=0.),
        
        A.Normalize(mean=(0,0,0), std=(1,1,1)), # 标准化
        ToTensorV2(),
    ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})
    
test_transform = A.Compose([
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2(),
], additional_targets={'mask1': 'mask', 'mask2': 'mask'})

# MODIFIED: TRANSFORM
train_dataset = Task1Dataset(args.root_dir, split="train", transform=train_transform, target=args.target, size=args.img_size)
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_test = Task1Dataset(args.root_dir, split="test", transform=test_transform, target=args.target, size=args.img_size)
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    model = nn.DataParallel(model)

model = model.cuda()
model.train()
fc_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
save_interval = args.n_skip  # int(max_epoch/6)

iterator = tqdm(range(0, args.max_epochs), ncols=70)

iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.7
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.8, 0.999))
# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

for epoch in iterator:
    model.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)

        # CE loss
        loss_fc = fc_loss(outputs, label_batch[:].long())
        # Dice loss
        loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
        loss = loss_dice * 0.6 + loss_fc * 0.4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        #lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1

        logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss/len(train_dataset))

    # loss visualization

    # fig1, ax1 = plt.subplots(figsize=(11, 8))
    # ax1.plot(range(epoch + 1), Loss)
    # ax1.set_title("Average trainset loss vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Current loss")
    # plt.savefig('loss_vs_epochs_gauss.png')

    # plt.clf()
    # plt.close()

    if (epoch + 1) % save_interval == 0:
        avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir)
        # avg_dcs, avg_hd = test()
        
        if avg_dcs > Best_dcs:
            save_mode_path = os.path.join(args.save_path, 'epoch={}_lr={}_avg_dcs={}_avg_hd={}_target={}.pth'.format(epoch, lr_, avg_dcs, avg_hd, args.target))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            #temp = 1
            Best_dcs = avg_dcs

        Test_Accuracy.append(avg_dcs)

        # val visualization

        # fig2, ax2 = plt.subplots(figsize=(11, 8))
        # ax2.plot(range(int((epoch + 1) // save_interval)), Test_Accuracy)
        # ax2.set_title("Average val dataset dice score vs epochs")
        # ax2.set_xlabel("Epoch")
        # ax2.set_ylabel("Current dice score")
        # plt.savefig('val_dsc_vs_epochs_gauss.png')

        # plt.clf()
        # plt.close()

    if epoch >= args.max_epochs - 1:
        save_mode_path = os.path.join(args.save_path,  'epoch={}_lr={}_avg_dcs={}_target={}.pth'.format(epoch, lr_, avg_dcs, args.target))
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        iterator.close()
        break