#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch
from medpy import metric
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import SimpleITK as sitk
from scipy.ndimage import zoom

class Normalize():
    def __call__(self, sample):

        function = transforms.Normalize((.5 , .5, .5), (0.5, 0.5, 0.5))
        return function(sample[0]), sample[1]


class ToTensor():
    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        function = transforms.ToTensor()
        return function(sample[0]), function(sample[1])


class RandomRotation():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        random_angle = np.random.randint(0, 360)
        return img.rotate(random_angle, Image.NEAREST), label.rotate(random_angle, Image.NEAREST)


class RandomFlip():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        temp = np.random.random()
        if temp > 0 and temp < 0.25:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        elif temp >= 0.25 and temp < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)
        elif temp >= 0.5 and temp < 0.75:
            return img.transpose(Image.ROTATE_90), label.transpose(Image.ROTATE_90)
        else:
            return img, label


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

'''
def calculate_metric_percase(output, target):
    smooth = 1e-5  

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if output.sum() > 0 and target.sum() > 0:
        hd = metric.binary.hd(output, target)
    else:
        hd = 0
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth), hd
'''
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def repaint(img):
    new_img = np.zeros((3, 512, 512))
    new_img[0] = np.where(img[0] == 1, 255, 0)
    new_img[1] = np.where(img[0] == 2, 255, 0)
    return new_img.astype(np.uint8).transpose(1,2,0)

def vote(img, classes):
    out = np.zeros((1, img.shape[1], img.shape[2]))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            out[0, i, j] = np.argmax(np.bincount(img[:, i, j]))
    return out

def test_single_volume(image, label, net, classes, patch_size=[512, 512], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(label.shape[0]):
            slice = image[ind, :, :]
            label_slice = label[ind, :, :]

            #imshow(slice, "./out/" + case[:-4] + "_img_" + str(ind) + ".jpg", denormalize=False)
            #imshow(label_slice, "./out/" + case[:-4] + "_label_" + str(ind) + ".jpg", denormalize=False)

            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            input_90 = torch.from_numpy(np.rot90(slice, 1).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input_180 = torch.from_numpy(np.rot90(slice, 2).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input_270 = torch.from_numpy(np.rot90(slice, 3).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input_f = torch.from_numpy(np.flip(slice).copy()).unsqueeze(0).unsqueeze(0).float().cuda()
            input = torch.stack([ input[0], input_90[0], input_180[0], input_270[0], input_f[0] ])
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out[1] = np.rot90(out[1], -1)
                out[2] = np.rot90(out[2], -2)
                out[3] = np.rot90(out[3], -3)
                out[4] = np.flip(out[4])
                out = vote(out, classes)

                #imshow(out,  "./out_1/" + case[:-4] + "_pre_" + str(ind) + ".jpg", denormalize=False)

                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        Image.fromarray((image * 255).astype(np.uint8).transpose(1,2,0), mode="RGB").save(test_save_path + '/' + case[0] + "_img.png")
        Image.fromarray(repaint(prediction), mode="RGB").save(test_save_path + '/' + case[0] + "_pred.png")
        Image.fromarray(repaint(label), mode="RGB").save(test_save_path + '/' + case[0] + "_gt.png")
        #img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        #prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        #lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        #img_itk.SetSpacing((1, 1, z_spacing))
        #prd_itk.SetSpacing((1, 1, z_spacing))
        #lab_itk.SetSpacing((1, 1, z_spacing))
        #sitk.WriteImage(prd_itk, test_save_path + '/'+ case + "_pred.nii.gz")
        #sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        #sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list