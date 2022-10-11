#========================================================================
#       MNIST_SIMPLE
#========================================================================

#Imports
import os
import torch
import torch.nn as nn       #All neutral network modules nn.Linear,nn.Conv2d,BatchNorm,Loss functions
import torch.optim as optim     #For all Optimization algorithms,SGD,Adam,etc
import torch.nn.functional as F    #All function that don't have any parameters
from torch.utils.data import DataLoader     #Gives easizer dataset managment and creates mini batches
import torchvision.datasets as datasets     #Has standard datasets we can import in a nice way
import torchvision.transforms as transforms     #矩阵转置

#Hyperparameters
in_channel = 1      #CNN输入通道为1
input_size = 784        #输入28x28
num_classes = 10        #输出大小
learning_rate = 1e-4       #学习率
batch_size = 1024     #批大小
num_epochs = 10      #循环次数
load_model = True
fname = 'my_checkpoint.pth.tar'

#Create fully connected nerwork
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)     #nn.Linear 用于二维全链接
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):        #向前
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#TODO:Creat simple CNN
class CNN(nn.Module):
    def __init__(self,in_channel = 1,num_classes = 10):#本次读取手写，只有黑白色，所以一通道，如果是彩色，应该是三通道
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))#卷积层
        #kernel_size 内核大小 stride 步幅大小 padding 填充
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x
#保存状态
def save_checkpoint(state,filename=fname):
    print("=> Saving checkpoint")
    torch.save(state,filename)

#下载状态
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load Data

#划分数据集
#train_dataset,test_dataset = torch.utils.data.random_split(dataset,[n,m]) #n,m为具体数值
#train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
#test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)


train_dataset = datasets.MNIST(root="MNIST_data/",train=True,transform=transforms.ToTensor(),download=True)
#root 下载数据集保存地址 train 是否为训练集 transforms 下载的为numpy型，转化为张量 download = true 下载数据集
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
#shuffle=True 每次打乱数据集
test_dataset = datasets.MNIST(root="MNIST_data/",train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)

#Initialize network
# model = NN(input_size=input_size,num_classes=num_classes).to(device)      #用NN简单模型
model = CNN().to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()       #loss函数
optimizer = optim.Adam(model.parameters(),lr=learning_rate)     #使用Adam优化
#model.parameters() 模型保存的参数

#下载上次的状态
if os.path.exists(fname):       #os.path.exists 检查文件是否存在
    load_checkpoint(torch.load(fname))

#Train Network
for epoch in range(num_epochs):
    losses = []

    #每三次保存一次状态
    if epoch % 3 ==0:
        checkpoint = {'state_dict' : model.state_dict(),'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx,(data,targets) in enumerate(train_loader):        #enumerate 将列表变成有索引的字典
        #Get data to cuda if possible
        data = data.to(device = device)     #传给gpu/cpu
        targets = targets.to(device=device)

        # 使用NN模型时开启
        # #Get to correct shape
        # data = data.reshape(data.shape[0],-1)       #-1保证压平

        #forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())

        #backward
        optimizer.zero_grad()       #梯度清零
        loss.backward()

        #gradient descent or adam step
        optimizer.step()

    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')

#Check accuracy on training & test to see how good our model

# def check_accuracy(loader,model):
#     num_correct = 0     #记录输出结果
#     num_samples = 0     #记录样品个数
#     model.eval()

#     with torch.no_grad():
#         for x,y in loader:
#             x = x.to(device=device)     #x是data
#             y = y.to(device=device)     #y是label
#             #使用NN模型时开启
#             # x = x.reshape(x.shape[0],-1)

#             scores = model(x)
#             _,predictions = scores.max(1)       #在第一维度（看每一行）最大值
#             num_correct += (predictions == y).sum()     #看索引即可
#             num_samples += predictions.size(0)

#         print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / num_samples *100:.2f}')

#     model.train()

# check_accuracy(train_loader,model)
# check_accuracy(test_loader,model)