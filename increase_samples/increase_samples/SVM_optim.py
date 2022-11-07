#支持向量机回归作为替代模型
#使用MAE作为损失函数查看训练集表现
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    #输入四个参数，输出1个s参数
    path = r'total_data.csv'
    input_dim = 4
    hidden_dim = 8
    output_dim = 1

    SVM_model = svm.SVR()

    data = pd.read_csv(path,index_col=False,header=None,dtype=np.float32)
    data_total = np.array(data)

    #划分数据
    data_input = data_total[:,1:5]
    data_output = data_total[:,5]
    train_input, test_input, train_output, test_output = train_test_split(data_input,data_output,test_size=0.5)

    #训练
    SVM_model.fit(train_input,train_output)

    #检查
    pred = SVM_model.predict(test_input)

    loss = []
    for i in range(len(test_input)):
        #print(test_output[i],'------',pred[i])
        loss.append(abs(test_output[i]-pred[i]))
    print(max(loss))
    #绘图
    #px = np.array(range(len(test_loss)))
    #ptrain_y = np.array(train_loss)
    #ptest_y = np.array(test_loss)
    #plt.plot(px,ptrain_y,color='red')
    #plt.plot(px,ptest_y,color='green')
    #plt.show()