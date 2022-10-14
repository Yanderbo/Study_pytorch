#imports
from cProfile import label
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import CatAndDogsDataset

#Load Data
my_transforms = transforms.Compose([
        transforms.ToPILImage(),#是转换数据格式，把数据转换为tensfroms格式
        transforms.Resize((256,256)),#扩大为256，若只有一个变量，则短的变成256，另一个等比扩大
        transforms.RandomCrop((224,224)),#剪裁为224
        #transforms.RandomResizedCrop(224,scale=(0.5,0.1))，#剪裁大小，及中心
        transforms.ColorJitter(brightness=0.5),#修改亮度，对比度等
        transforms.RandomRotation(degrees=45),#旋转
        transforms.RandomHorizontalFlip(p=0.5),#水平翻转
        transforms.RandomVerticalFlip(p=0.05),#垂直翻转
        transforms.RandomGrayscale(p=0.2),#转灰度图
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
        #transforms.Normalize(mean=[0,0,0],std=[1,1,1])#(value - mean)/std
    ])

dataset = CatAndDogsDataset(csv_file="cats_dogs/cats_dogs.csv",
                            root_dir="cats_dogs/cats_and_dogs",
                            transform=my_transforms)

img_num = 0
for img,label in dataset:
    save_image(img,'img'+str(img_num)+'.png')
    img_num += 1