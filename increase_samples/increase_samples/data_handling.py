import pandas as pd
import numpy as np
import os

data_path = r'1000samples'
parameters_path = r'parameters.csv'
para_num = 4
data_num = len(os.listdir(data_path))#数据个数
parameter = pd.read_csv(parameters_path,header=None)
s11_data = pd.DataFrame(np.zeros((data_num,para_num),dtype=np.float32))#用于记录汇总新数据
data_new = pd.DataFrame(np.zeros((data_num,6)))#用于构建新的数据集

#整理文件
for i in np.arange(1,data_num+1):
        #数据文件操作
        current_path = data_path+'\\S_1000samples%s.s1p'%i
        current_data = pd.read_csv(current_path)
        current_data.drop(list(range(0, 8)), inplace=True)#删除头部字符串行，一行数据被读取到一列中
        data_savepath = r'data_handled\S_1000samples%s.s1p'%i
        current_data.to_csv(data_savepath, index=False)#将处理后的保存至原地址
        current_data = pd.read_csv(data_savepath, names=['Frequency', 'S11', 'R'],
                              index_col=False, delim_whitespace=True)#再次读取文件，这次可以使数据被正确读取，分开至不同的列

        data_new.iloc[i-1][1:5] = parameter.iloc[i-1][0:4]
        data_new.iloc[i-1][0] = current_data.iloc[51][0]
        data_new.iloc[i-1][5] = current_data.iloc[51][1]




totaldata_save_path = r'total_data.csv'
data_new.to_csv(totaldata_save_path,index=False)