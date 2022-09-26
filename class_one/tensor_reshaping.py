import torch

#========================================================================
#       Tensor Reshaping
#========================================================================

x = torch.arange(9)

x_3x3 = x.view(3,3)     #将x重新规划城3x3的矩阵
# x_3x3 = x.reshape(3,3)

y = x_3x3.t()       #转置
print(y.contiguous().view(9))
# print(y.reshape(9))

#view注重连续性，需要在空间上连续
#reshape均可

x1 = torch.rand([2,5])
x2 = torch.rand([2,5])
print(torch.cat((x1,x2),dim=0).shape)
# out:torch.Size([4, 5])
print(torch.cat((x1,x2),dim=1).shape)
# out:torch.Size([2, 10])
#注意维度问题

z = x1.view(-1,2)       #-1的意义，自动匹配view.(-1,2)即每行两个，自动分行
print(z.shape)

#eg
batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)


x = torch.tensor([[[1,2,3],
                  [1,2,3]],
                  [[1,2,3],
                  [1,2,3]]])
z = x.permute(0,2,1)        #起到转置的作用
print(z)
# out：tensor([[[1, 1],
#          [2, 2],
#          [3, 3]],

#         [[1, 1],
#          [2, 2],
#          [3, 3]]])

x= torch.arange(10)
print(x.shape)
#out:torch.Size([10])
print(x.unsqueeze(0).shape)     #在第0维度上填充,行上
#out:torch.Size([1, 10])
print(x.unsqueeze(1).shape)     #在第1维度上填充，列上
#out:torch.Size([10, 1])

x =torch.arange(10).unsqueeze(0).unsqueeze(1)
#out:torch.Size([1, 1, 10])

print(x.squeeze(1).shape)       #删除某维度