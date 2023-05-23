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
from dataset.dataset_Miccai import Task1Validate, Task1Dataset
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
from PIL import Image
import SimpleITK as sitk

def repaint(img):
    new_img1 = np.zeros((512, 512))
    new_img2 = np.zeros((512, 512))
    new_img1 = np.where(img[0] == 1, 1, 0)
    new_img2 = np.where(img[0] == 2, 1, 0)
    return new_img1.astype(np.uint8), new_img2.astype(np.uint8)

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
parser.add_argument("--target", default=0, type=int)
parser.add_argument("--test", default=False, type=bool)
args = parser.parse_args()

if args.target == 1:
    args.img_size = 512
    args.num_classes = 2
else:
    args.img_size = 512
    args.num_classes = 3

eval_save_path = "./eval"
model=MTUNet(args.num_classes, args.img_size) # 9

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

import albumentations as A
from albumentations.pytorch import ToTensorV2
    
test_transform = A.Compose([
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2(),
], additional_targets={'mask1': 'mask', 'mask2': 'mask'})

# MODIFIED: TRANSFORM
if args.test:
    validate_dataset = Task1Dataset(args.root_dir, split="test", transform=test_transform, target=args.target, size=args.img_size)
else:
    validate_dataset = Task1Validate(args.root_dir, split="validate", transform=test_transform, target=args.target, size=args.img_size)
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True)

if args.n_gpu > 1:
    model = nn.DataParallel(model)


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

def vote(img, classes):
    out = np.zeros((1, img.shape[1], img.shape[2]))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            out[0, i, j] = np.argmax(np.bincount(img[:, i, j]))
    return out

model = model.cuda()
model.eval()
metric_list = 0.0
if args.test:
    perf, hd = inference(args, model, validate_loader, args.test_save_dir)
else:
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(validate_loader)):
            h, w = sampled_batch["image"].size()[2:]
            image, case_name = sampled_batch["image"], sampled_batch["case_name"]
            image = image.squeeze(0).cpu().detach().numpy()
            prediction = np.zeros((1, 512, 512))
            ind = 0
            print(image.shape)
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            input_90 = torch.from_numpy(np.rot90(slice, 1).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input_180 = torch.from_numpy(np.rot90(slice, 2).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input_270 = torch.from_numpy(np.rot90(slice, 3).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input_f = torch.from_numpy(np.flip(slice).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input = torch.stack([ input[0], input_90[0], input_180[0], input_270[0], input_f[0] ])
            outputs = model(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out[1] = np.rot90(out[1], -1)
            out[2] = np.rot90(out[2], -2)
            out[3] = np.rot90(out[3], -3)
            out[4] = np.flip(out[4])
            out = vote(out, args.num_classes)

            #imshow(out,  "./out_1/" + case[:-4] + "_pre_" + str(ind) + ".jpg", denormalize=F
            prediction[ind] = out
            
            if args.target == 1:
                img = prediction.astype(np.uint8)[0]
                img = Image.fromarray(img, mode="L").resize((1024,1024), Image.NEAREST)
                img.save(eval_save_path + '/2/' + case_name[0])
            else:
                img1, img3 = repaint(prediction)
                Image.fromarray(img1 * 255, mode="L").save('test2/1/' + case_name[0])
                Image.fromarray(img3 * 255, mode="L").save('test2/3/' + case_name[0])
                img1 = Image.fromarray(img1, mode="L").resize((1024,1024), Image.NEAREST)
                img3 = Image.fromarray(img3, mode="L").resize((1024,1024), Image.NEAREST)
                img1.save(eval_save_path + '/1/' + case_name[0])
                img3.save(eval_save_path + '/3/' + case_name[0])