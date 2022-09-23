import torch

#========================================================================
#       Tensor Math & Comparison Operations
#========================================================================

x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

#Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)       #x+y输出保存在z1

z2 = torch.add(x,y)
z = x + y       #常用

#Subtraction
z = x - y

#Dibision
z = torch.true_divide(x,y)      #true_divide()而不是divide()

#Inplace operations     #原地变化
t = torch.empty(3)
t.add_(x)       #x加到t上
t += x

#Exponentiation
z = x.pow(2)        #x每个元素二次方
z = x.pow(y)        #x第n个元素对应y第n个元素的次方
# print(z)
z = x ** 2      #常用

#Simple comparison
z = x > 0
z = x < 0       #返回由True False组成的张量

#Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)        #x3.shape=(2,3)
x3 = x1.mm(x2)      #mm二维元素

#Matirx exponentiation
matrix_exp = torch.rand(5,5)
#print(matrix_exp.matrix_power(3))      #矩阵自身乘自身3次(矩阵的幂)

#Element wise mult
z = x * y
# print(z)

#dot product
z = torch.dot(x,y)      #仅限一维元素，将所有加在一起

#Batch Matirx Multilication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2)    #(batch,n,p) 三维运算

#Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

#Other useful tensor operation

#一维时
#x0 = torch.tensor([1,2,3])
#sum_x0 = torch.sum(x0,dim=0)        #一维时，dim=0，列加列（行总和）
# out:tensor(6)

#二维时
#x0 = torch.tensor([[1,2,3],
#                   [4,5,6]])

#sum_x0 = torch.sum(x0,dim= 0)       #二维时，dim=0,行加行(列总和)
# out:tensor([5, 7, 9])
#sum_x0 = torch.sum(x0,dim= 1)       #二维时，dim=1，列加列（行总和）
# out:tensor([ 6, 15])

# 三维时
x0 = torch.tensor([[[1,2,3],
                    [4,5,6]],
                    [[1,2,3],
                    [4,5,6]]])

#sum_x0 = torch.sum(x0,dim= 0)       #三维时dim=0 层加层，按照层的方向相加
#out:tensor([[ 2,  4,  6],
#            [ 8, 10, 12]])
#sum_x0 = torch.sum(x0,dim= 1)       #三维时dim=1 行加行，按照行的方向相加
# out:tensor([[5, 7, 9],
#             [5, 7, 9]])
#sum_x0 = torch.sum(x0,dim= 2)       #三维时dim=2 列加列，按照列的方向相加
# out:tensor([[ 6, 15],
#             [ 6, 15]])
#print(sum_x0)

values,indices = torch.max(x0,dim=0)     #values 为最大值，indices 为最大值在的位置
values,indices = torch.min(x0,dim=0)     ##values 为最小值，indices 为最小值在的位置
abs_x0 = torch.abs(x0)      #取绝对值
indices = torch.argmax(x0,dim=0)        #返回值为最大值位置
indices = torch.argmin(x0,dim=0)        #返回值为最小值位置
mean_x0 = torch.mean(x0,dim=0)      #某方向平均值
z = torch.eq(x,y)       #判断相应位置是否相同
