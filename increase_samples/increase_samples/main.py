#深度神经网络拟合数据
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#训练过程函数
def train_loop(dataloader, model, loss_fn, optimizer, train_loss):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        y = torch.reshape(y,(50,1))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            train_loss.append(loss)

#测试过程函数
def test_loop(dataloader, model, loss_fn, test_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            y = torch.reshape(y,(-1, 1))
            pred = model(X)
            loss = loss_fn(pred,y)
            test_loss.append(loss)
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
                               nn.Linear(10,100),nn.Softplus(),
                               nn.Linear(100, 100), nn.Softplus(),
                               nn.Linear(100, 10), nn.Softplus(),
                               nn.Linear(10,output_dim))

    data = pd.read_csv(path,index_col=False,header=None,dtype=np.float32)
    data_total = np.array(data)

    #划分数据
    data_input = data_total[:,1:5]
    data_output = data_total[:,5]
    train_input, test_input, train_output, test_output = train_test_split(data_input,data_output,test_size=0.5)

    #转化为tensor并进行归一化
    train_input = torch.from_numpy(train_input).type(torch.float32)
    train_output = torch.from_numpy(train_output).type(torch.float32)

    test_input = torch.from_numpy(test_input).type(torch.float32)
    test_output = torch.from_numpy(test_output).type(torch.float32)


    #进行归一化处理
    train_input = torch.nn.functional.normalize(train_input,p=2,dim=1)
    test_input = torch.nn.functional.normalize(test_input,p=2,dim=1)

    # batch size和迭代次数
    batch_size = 50
    epochs = 200

    #合并
    train = torch.utils.data.TensorDataset(train_input, train_output)
    test = torch.utils.data.TensorDataset(test_input, test_output)

    #转化为dataloader
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=500, shuffle=True)

    loss_func = nn.MSELoss()
    learning_rate = 0.05
    optimizer = torch.optim.Adam(ANN_model.parameters(), lr=learning_rate)
    test_loss = []
    train_loss = []

    for i in range(epochs):
        train_loop(train_loader, ANN_model, loss_func, optimizer, train_loss)
        test_loop(test_loader, ANN_model, loss_func, test_loss)
        print('---------------------------------')
        print('epoch:%s'%i)

    print("Done!")


    #绘图
    px = np.array(range(len(test_loss)))
    #ptrain_y = np.array(train_loss)
    ptest_y = np.array(test_loss)
    #plt.plot(px,ptrain_y,color='red')
    plt.plot(px,ptest_y,color='green')
    plt.show()


















