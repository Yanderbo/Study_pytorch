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
x3 = x1.mm(x2)

#Matirx exponentiation
matrix_exp = torch.rand(5,5)
#print(matrix_exp.matrix_power(3))      #矩阵自身乘自身3次(矩阵的幂)

#Element wise mult
z = x * y
# print(z)