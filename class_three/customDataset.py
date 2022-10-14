#Imports
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from torchvision.utils import save_image

class CatAndDogsDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv(csv_file)#忽略第一行，所以第一行加种类
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)       #具体数量

    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = cv2.imread(img_path)#读取照片
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image,y_label)