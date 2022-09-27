#========================================================================
#       MNIST_SIMPLE
#========================================================================

#Imports
from traceback import print_tb
import torch
import torch.nn as nn       
import torch.optim as optim     #优化算法，一些损失函数
import torch.nn.functional as F    #像relu等函数
from torch.utils.data import DataLoader     
import torchvision.datasets as datasets     #含有一些数据集
import torchvision.transforms as transforms     #矩阵转置

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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784        #输入28x28
num_classes = 10        #输出大小
learning_rate = 0.001       #学习率
batch_size = 64     #批大小
num_epochs = 1      #循环次数

#Load Data
train_dataset = datasets.MNIST(root="MNIST_data/",train=True,transform=transforms.ToTensor(),download=True)
#root 下载数据集保存地址 train 是否为训练集 transforms 下载的为numpy型，转化为张量 download = true 下载数据集
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
#shuffle=True 每次打乱数据集
test_dataset = datasets.MNIST(root="MNIST_data/",train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)

#Initialize network
model = NN(input_size=input_size,num_classes=num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()       #loss函数
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#model.parameters() 模型保存的参数

#Train Network

for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        #Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device=device)

        #Get to correct shape
        data = data.reshape(data.shape[0],-1)       #-1保证压平

        #forward
        scores = model(data)
        loss = criterion(scores,targets)

        #backward
        optimizer.zero_grad()       #梯度清零
        loss.backward()

        #gradient descent or adam step
        optimizer.step()

#Check accuracy on training & test to see how good our model

def check_accuracy(loader,model):
    num_correct = 0     #记录输出结果
    num_samples = 0     #记录样品个数
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)     #x是data
            y = y.to(device=device)     #y是label
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,predictions = scores.max(1)       #在第一维度（看每一行）最大值
            num_correct += (predictions == y).sum()     #看索引即可
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / num_samples *100:.2f}')

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)