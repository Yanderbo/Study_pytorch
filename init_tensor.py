import torch

#===========================================
#       Initializing Tensor
#===========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

#建立已知数据的张量
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,
                                            device=device,requires_grad=True)

#other common initialization methods
x = torch.empty(size=(3,3))     #生成空的矩阵
x = torch.ones((3,3))   #生成全是1的矩阵
x = torch.zeros((3,3))  #生成全是0的矩阵
x = torch.rand((3,3))   #生成0~1的正态分布数
x = torch.eye(3,3)    #生成单位矩阵
x = torch.arange(start=0,end=5,step=0.5)  #从0开始，每次加1，到5为止（但不要5）
x = torch.linspace(start=0.1,end=1,steps=10)    #从0.1开始，到1结束，均分10步（有0.1，有1）
x = torch.empty(size=(3,3)).normal_(mean=0,std=1)       #生成均值为0，方差为1，的正太分布数
x = torch.empty(size=(3,3)).uniform_(0,1)       #生成均匀分布
print(x)

