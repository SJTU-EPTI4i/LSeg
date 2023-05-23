import os
import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class Task1Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None, target=0, size=256):
        super().__init__()
        
        self.data_dir = data_dir
        self.transform = transform

        self.target = int(target)
        
        self.task_tag = 'A. Segmentation'
        self.split = split
        if split == "train":
            df = pd.read_csv(os.path.join(data_dir, 'segmentation_split.csv'))
        else:
            df = pd.read_csv(os.path.join(data_dir, 'segmentation_split_test.csv'))

        if self.target == 1:
            odf = df
            df = pd.DataFrame({ "filename" : [ d for d in odf['filename'] if len(d) == 7 ] })

        self.df = df
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.iloc[index]
        '''
        slice_pos = index % 4
        if slice_pos == 0:
            x1, x2, y1, y2 = 0, 512, 0, 512
        elif slice_pos == 1:
            x1, x2, y1, y2 = 512, 1024, 0, 512
        elif slice_pos == 2:
            x1, x2, y1, y2 = 0, 512, 512, 1024
        else:
            x1, x2, y1, y2 = 512, 1024, 512, 1024
        '''

        # 图像
        filename = info['filename']
        img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.size, self.size))
        #if self.target == 1:
        #    img = cv2.resize(img, (self.size, self.size))
        #else:
        #    img = img[x1:x2, y1:y2]
        #    img = cv2.resize(img, (self.size, self.size))
        
        # 标签
        lbl = []
        for c in ['1. Intraretinal Microvascular Abnormalities', '2. Nonperfusion Areas', '3. Neovascularization']:
            lbl_path = os.path.join(self.data_dir, 'A. Segmentation', '2. Groundtruths', 'a. Training Set', c, filename)
            if os.path.exists(lbl_path):
                mask = cv2.imread(lbl_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                assert (np.unique(mask) == [0, 255]).all(), np.unique(mask)
                mask = cv2.resize(mask, (self.size, self.size))
                #if self.target == 1:
                #    mask = cv2.resize(mask, (self.size, self.size))
                #else:
                #    mask = mask[x1:x2, y1:y2]
                #    mask = cv2.resize(mask, (self.size, self.size))
                lbl.append((mask/255.).astype(np.float))
            else:
                #lbl.append(np.zeros((1024,1024),np.float))
                lbl.append(np.zeros((self.size,self.size),np.float))
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl[0], mask1=lbl[1], mask2=lbl[2])
            img = aug['image']
            lbl[0] = aug['mask']
            lbl[1] = aug['mask1']
            lbl[2] = aug['mask2']

        #Image.fromarray((lbl[-1] * 255).astype(np.uint8), mode="L").save(f'./test/test_lbl_{c}.png')
        #print((img.numpy() * 255).astype(np.uint8))
        #Image.fromarray((img.numpy() * 255).astype(np.uint8).transpose(1,2,0), mode="RGB").save(f'./test/test_img.png')
        #input()
        #lbl = torch.stack(lbl)

        # 子任务2
        if self.target == 1:
            return {'image': img, 'label': lbl[1], 'case_name': filename}
        # 子任务1/3, 其中子任务1的标签设置为1,3的标签设置为2
        return {'image': img, 'label': lbl[0] + lbl[2] * 2, 'case_name': filename}


class Task1Validate(Dataset):
    def __init__(self, data_dir, split, transform=None, target=0, size=256):
        super().__init__()
        
        self.data_dir = data_dir
        self.transform = transform

        self.target = int(target)
        
        self.task_tag = 'A. Segmentation'
        self.split = split
        tdir = "data/Miccai/A. Segmentation/1. Original Images/b. Testing Set/"
        self.files = os.listdir(tdir)
        #self.df = df[df['split'] == split]
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]

        # image
        img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'b. Testing Set', filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.size, self.size))
        #if self.target == 1:
        #    img = cv2.resize(img, (self.size, self.size))
        #else:
        #    img = img[x1:x2, y1:y2]
        #    img = cv2.resize(img, (self.size, self.size))
        
        if self.transform is not None:
            aug = self.transform(image=img)
            img = aug['image']

        #Image.fromarray((lbl[-1] * 255).astype(np.uint8), mode="L").save(f'./test/test_lbl_{c}.png')
        #print((img.numpy() * 255).astype(np.uint8))
        #Image.fromarray((img.numpy() * 255).astype(np.uint8).transpose(1,2,0), mode="RGB").save(f'./test/test_img.png')
        #input()
        #lbl = torch.stack(lbl)
        print(img.shape)
        return {'image': img, 'case_name': filename}
