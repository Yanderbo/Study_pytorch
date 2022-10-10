#========================================================================
#       MNIST_SIMPLE
#========================================================================

#Imports
import torch
import torch.nn as nn       #All neutral network modules nn.Linear,nn.Conv2d,BatchNorm,Loss functions
import torch.optim as optim     #For all Optimization algorithms,SGD,Adam,etc
import torch.nn.functional as F    #All function that don't have any parameters
from torch.utils.data import DataLoader     #Gives easizer dataset managment and creates mini batches
import torchvision.datasets as datasets     #Has standard datasets we can import in a nice way
import torchvision.transforms as transforms     #矩阵转置

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2      #RNN层数
hidden_size = 256       #隐藏层
num_classes = 10        #输出大小
learning_rate = 0.001       #学习率
batch_size = 64     #批大小
num_epochs = 2      #循环次数

# #Creat a RNN
# class RNN(nn.Module):
#     def __init__(self,input_size,hidden_size,num_layers,num_classes):
#         super(RNN,self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
#         self.fc = nn.Linear(hidden_size*sequence_length,num_classes)

#     def forward(self,x):
#         h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)

#         out, _ = self.rnn(x,h0)
#         out = out.reshape(out.shape[0],-1)
#         out = self.fc(out)
#         return out

#Creat a bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)

        out, _ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])

        return out

#Load Data
train_dataset = datasets.MNIST(root="MNIST_data/",train=True,transform=transforms.ToTensor(),download=True)
#root 下载数据集保存地址 train 是否为训练集 transforms 下载的为numpy型，转化为张量 download = true 下载数据集
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
#shuffle=True 每次打乱数据集
test_dataset = datasets.MNIST(root="MNIST_data/",train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)

#Initialize network
# model = NN(input_size=input_size,num_classes=num_classes).to(device)      #用NN简单模型
model = BRNN(input_size,hidden_size,num_layers,num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()       #loss函数
optimizer = optim.Adam(model.parameters(),lr=learning_rate)     #使用Adam优化
#model.parameters() 模型保存的参数

#Train Network

for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):        #enumerate 将列表变成有索引的字典
        #Get data to cuda if possible
        data = data.squeeze(1)
        data = data.to(device = device)    #传给gpu/cpu
        targets = targets.to(device=device)

        # 使用NN模型时开启
        # #Get to correct shape
        # data = data.reshape(data.shape[0],-1)       #-1保证压平

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
            x = x.to(device=device).squeeze(1)     #x是data
            y = y.to(device=device)     #y是label
            #使用NN模型时开启
            # x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,predictions = scores.max(1)       #在第一维度（看每一行）最大值
            num_correct += (predictions == y).sum()     #看索引即可
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / num_samples *100:.2f}')

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)