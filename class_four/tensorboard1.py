from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

#/disk2/zhangyanbo/Study_pytorch/class_three/cats_dogs/cats_and_dogs/cat.1.jpg
cat_img = Image.open('../class_three/cats_dogs/cats_and_dogs/cat.1.jpg')
print(f'old size {cat_img.size}')
transform_224 = transforms.Compose([
        transforms.Resize(224), # 这里要说明下 Scale 已经过期了，使用Resize
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
cat_img_224=transform_224(cat_img)
writer = SummaryWriter(log_dir='./logs', comment='cat image') # 这里的logs要与--logdir的参数一样
writer.add_image("cat",cat_img_224)
writer.close()# 执行close立即刷新，否则将每120秒自动刷新

x = torch.FloatTensor([100])
y = torch.FloatTensor([500])

for epoch in range(30):
    x = x * 1.2
    y = y / 1.1
    loss = np.random.random()
    with SummaryWriter(log_dir='./logs', comment='train') as writer: #可以直接使用python的with语法，自动调用close方法
        writer.add_histogram('his/x', x, epoch)
        writer.add_histogram('his/y', y, epoch)
        writer.add_scalar('data/x', x, epoch)
        writer.add_scalar('data/y', y, epoch)
        writer.add_scalar('data/loss', loss, epoch)
        writer.add_scalars('data/data_group', {'x': x,
                                                'y': y}, epoch)