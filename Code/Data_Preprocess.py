# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 22:23:57 2021

@author: Gorden Li
"""
import pandas as pd
import numpy as np

def readasdf(route):
    csv_file = route
    csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
    df = pd.DataFrame(csv_data)
    df = df.loc[ : , ~df.columns.str.contains("^Unnamed")]#读取列名非'Unnamed'的列
    return(df)

def addblankcol(df, index, colnamelist):
    for colname in colnamelist:
        df.insert(index, colname, '')
        index += 1
    return(df)

def split_train_test(data, test_ratio, seed):
    #设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(seed)
    #permutation随机生成[0, len(data))随机序列
    shuffled_indices = np.random.permutation(len(data)) #generate a list of random with no replacement
    #test_ratio为测试集所占的半分比
    test_set_size = int(round(len(data) * test_ratio, 0))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    #iloc选择参数序列中所对应的行
    return data.iloc[train_indices], data.iloc[test_indices]   


##Import data
data = readasdf('Training Data for Ag Project.csv')
data.columns = data.columns.str.replace('-','_').str.replace(' ','_')
data = addblankcol(data, 18, ['Temp','Prec','Rad'])
data.info()

##Fill column Temp, Prec, Rad Based on year
for row in range(len(data)):
    year_num = str(data.loc[row, 'GrowingSeason'])[-2:]
    for col in ['Temp','Prec','Rad']:
        data.loc[row, col] = data.loc[row, col + '_' + year_num]

for para in ['Temp','Prec','Rad']:
    for year in ['_03','_04','_05','_06','_07','_08','_09']:
        col_to_del = para + year
        data.drop([col_to_del], axis = 1, inplace = True)

data.to_csv('data.csv')




