import torch

#========================================================================
#       Tensor Indexing
#========================================================================

batch_size = 10
features = 25
x = torch.rand((batch_size,features))

# print(x[0,:])       #x[n,:]输出第n行，从零开始
# print(x[:,0])       #x[:,n]输出第n列，从零开始
# print(x[0,0:10])        #x[n,k:j]输出第n行，从k到j-1
# print(x[0:5,0])         #x[k:j,n]输出第n列，从k到j-1

#Fancy indexing
x = torch.arange(10)
indices = [2,5,8]
# print(x[indices])           #输出x列表索引为2，5，8的数

x = torch.tensor([[1,2,3,4,5],
                  [1,2,3,4,5],
                  [1,2,3,4,5]])
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
#print(x[rows,cols])         #输出[x[rows[0],cols[0],x[rows[],cols[1]]]

#More advanced indexing
x = torch.arange(10)
# print(x[(x<2) | (x>8)])     #输出x列表中严格小于2 或者 大于8的数
# print(x[(x>2) & (x<8)])     #输出x列表中严格大于2 并且 小于8的数
# print(x[x.remainder(3) == 0])       #输出x列表中除三余数为0的数

#Useful operations
print(torch.where(x>5,x,x*2))       #列表x中 大于5的不变，其余的变为二倍
print(torch.tensor([0,0,0,1,1,2,3,3,4]).unique())       #输出不带重复的列表
print(x.ndimension())       #输出x的维度
print(x.numel())        #输出x共有多少个元素      


