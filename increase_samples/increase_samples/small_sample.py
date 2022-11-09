#深度神经网络拟合数据
#本代码使用小样本进行训练，共取样本150个，100个作为训练，25个作为测试，25个作为验证
#使用MSE作为训练时的损失函数，MAE作为验证集的损失函数
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#训练过程函数
def train_loop(dataloader, model, loss_fn, optimizer, train_loss):
    size = len(dataloader.dataset)
    num_batchsize = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        y = torch.reshape(y,(-1 ,1))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        #if batch % num_batchsize == 0:
        #    train_loss.append(loss)

#测试过程函数
def test_loop(dataloader, model, loss_fn, test_loss):
    size = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            y = torch.reshape(y,(-1, 1))
            pred = model(X)
            loss = loss_fn(pred,y)#计算MAE
            test_loss.append(loss.item())
            #显示损失：
            print('test_loss:',loss)


if __name__ == '__main__':
    #输入四个参数，输出1个s参数
    path = r'total_data.csv'
    input_dim = 4
    hidden_dim = 8
    output_dim = 1

    #模型:两个隐藏层
    ANN_model = nn.Sequential( nn.Linear(input_dim,10), nn.Tanh(),
                               nn.Linear(10,120),nn.Softplus(),
                               nn.Linear(120, 140), nn.Softplus(),
                               nn.Linear(140, 50), nn.Softplus(),
                               nn.Linear(50,output_dim))

    data = pd.read_csv(path,index_col=False,header=None,dtype=np.float32)
    data_total = np.array(data)

    #训练集&测试集
    data_input = data_total[0:125,1:5]
    data_output = data_total[0:125,5]
    #验证集
    valid_in = data_total[125:150,1:5]
    valid_out = data_total[125:150,5]
    #划分数据集
    train_input, test_input, train_output, test_output = train_test_split(data_input,data_output,test_size=0.2)

    #转化为tensor
    train_input = torch.from_numpy(train_input).type(torch.float32)
    train_output = torch.from_numpy(train_output).type(torch.float32)

    test_input = torch.from_numpy(test_input).type(torch.float32)
    test_output = torch.from_numpy(test_output).type(torch.float32)

    valid_in = torch.from_numpy(valid_in).type(torch.float32)
    valid_out = torch.from_numpy(valid_out).type(torch.float32)

    #进行归一化处理
    #train_input = torch.nn.functional.normalize(train_input,p=2,dim=1)
    #test_input = torch.nn.functional.normalize(test_input,p=2,dim=1)

    #验证集归一化
    #valid_in = torch.nn.functional.normalize(valid_in, p=2, dim=1)


    # batch size和迭代次数
    batch_size = 100
    epochs = 1000

    #合并
    train = torch.utils.data.TensorDataset(train_input, train_output)
    test = torch.utils.data.TensorDataset(test_input, test_output)
    valid = torch.utils.data.TensorDataset(valid_in,valid_out)

    #转化为dataloader
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=25, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid,batch_size=25,shuffle=True)

    loss_func = nn.MSELoss()#训练集损失函数
    test_loss_func = nn.L1Loss(reduce='mean')#验证集的评价函数

    learning_rate = 0.005
    optimizer = torch.optim.Adam(ANN_model.parameters(), lr=learning_rate)
    test_loss = []
    train_loss = []

    for i in range(epochs):
        train_loop(train_loader, ANN_model, loss_func, optimizer, train_loss)
        test_loop(test_loader, ANN_model, loss_func, test_loss)
        print('---------------------------------')
        print('epoch:%s'%i)

    print("Done!")


    #绘图:
    px = np.array(range(epochs))
    ptrain = np.array(train_loss)
    ptest = np.array(test_loss)
    plt.plot(px,ptrain,color='red')
    plt.plot(px,ptest,color='green')
    plt.legend('train loss','test loss')
    plt.show()









