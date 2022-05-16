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
data = readasdf('data.csv')
data.columns = data.columns.str.replace('-','_').str.replace(' ','_')
data = data.drop(['Commercial_Yield','Yield_Difference','Location_Yield'], axis = 1)

##Random Forest(Random Forest will Feature Filter automatically while doing regression)---Random Forest do not accept one-hot variable, Random Forest do not need to normalization
from sklearn.ensemble import RandomForestRegressor

#根据皮尔逊相关系数选择与要预测的属性列SalePrice相关性最高的10个属性[:11]，选出11个是因为应变量自己与自己的相关性最高，所以要将它去除故选择排序后的前11个属性，再去除应变量
data.info()
features_corr = data.corr()['Variety_Yield'].abs().sort_values(ascending=False)
features = ['Miles', 'FRUITS Gross Weight', 'PEAR', 'FRUITS Size', 'Route Highway Density', \
                    'VEGETABLES Gross Weight', 'FRUITS Item Gross Weight', 'FRUITS Quantity', 'APPLE', 'STONE FRUIT, MIXED', \
                    'VEGETABLES Quantity', 'VEGETABLES Item Gross Weight', \
                    'Year 2018', 'Year 2019', 'Year 2020', 'Year 2021', \
                    'Spring', 'Summer', 'Autumn', 'Winter']

#使用随机森林模型进行拟合的过程
data = data.loc[data.Variety == 'V102',]
X = data.drop(['Variety_Yield'], axis = 1)
Y = data['Variety_Yield']

feat_labels = X.columns

rf = RandomForestRegressor(n_estimators=100, max_depth = None, oob_score = True, random_state = 42) #n_estimators = 100表示森林中有100棵子树, RandomForestRegressor回归树, RandomForestClassifier分类树

#rf_pipe = Pipeline([('imputer', SimpleImputer(strategy = 'median')), ('standardize', StandardScaler()), ('rf', rf)])
forest = rf.fit(X, Y)
 #导入测试集，forest的接口score计算的是模型准确率accuracy/R^2, 这个score use Out-Of-Bag samples to estimate the R^2 on unseen data.
print(forest.oob_score_)
##Check the model performance------------------------------------------------------------------------------
#根据随机森林模型的拟合结果选择特征
importance = forest.feature_importances_

#np.argsort()返回待排序集合从下到大的索引值，[::-1]实现倒序，即最终imp_result内保存的是从大到小的索引值
imp_result = np.argsort(importance)[::-1][:15] #a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序。因为最后一位是-1, 倒是第二位是-2, 以此类推第一位的号码是-len(a)-1

#按重要性从高到低输出属性列名和其重要性
for i in range(len(imp_result)):
    print("%2d. %-*s %f" % (i + 1, 30, feat_labels[imp_result[i]], importance[imp_result[i]]))

#对属性列，按属性重要性从高到低进行排序
feat_labels_important = [feat_labels[i] for i in imp_result]
#绘制特征重要性图像, 根据纳入全部Feature的随机森林模型
import matplotlib.pyplot as plt
plt.title('Feature Importance')
plt.bar(range(len(imp_result)), importance[imp_result], color='lightblue', align='center')
plt.xticks(range(len(imp_result)), feat_labels_important, rotation=90)
plt.xlim([-1, len(imp_result)])
plt.tight_layout()
plt.show()

# Calculate the absolute errors
errors = abs(Y_Predict - Y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'In Cost.') # Print out the mean absolute error (mae)

#这里的测试后用预测值和实际值相差多少来评估 ，即MAPE指标（和实际值相差多少百分比的均值）来看一下它的效果如何
# Calculate mean absolute percentage error (MAPE)
x = errors / Y_test
mape = 100 * (errors / Y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Mape Error', round(np.mean(mape), 2), '%')
print('Accuracy:', round(accuracy, 2), '%.')

#Hyperparameter Adjustment
max_depths = [i for i in range(5, 21)]
n_estimators = [i for i in range(10, 61)]
OOB_scores = []
for i in range(len(n_estimators)):
    n_estimator = n_estimators[i]
    OOB_scores_of_the_n_estimator = []
    for j in range(len(max_depths)):
        max_depth = max_depths[j]
        rf_adj = RandomForestRegressor(n_estimators = n_estimator, max_depth = max_depth, oob_score = True, random_state = 42)
        forest_adj = rf_adj.fit(X_train, Y_train)
        OOB_scores_of_the_n_estimator.append(forest_adj.oob_score_)
    OOB_scores.append(OOB_scores_of_the_n_estimator)

Heatmap_data = pd.DataFrame(OOB_scores)
Heatmap_data.to_csv('C:/Personal/Wustl-MSCA/Spring 2021/Data Analysist Competition/Analysis/Code/Hyperparameter.csv')
#Heatmap of Hyperparameter Adjustment
import matplotlib.pyplot as plt
import seaborn as sns

#读取数据
file1= 'C:\Personal\Wustl-MSCA\Spring 2021\Data Analysist Competition\Analysis\Code\Hyperparameter_Heatmap.csv'

pt1=pd.read_csv(file1,index_col=u'OOB_Score')

#Picture1
#创造绘图区域，分两块：ax1与ax2。每块内部行列(a,b)未列明，由数据决定。figsize=(m,n)，为设置子图像长宽。nrows为图像个数
#f, (ax1,ax2) = plt.subplots(a,b,figsize = (5,5),nrows=2)---a,b为提前设置内部子图像个数
f, ax1 = plt.subplots(figsize = (8,3))

#设置数字与图像的转换，cmap用cubehelix map颜色
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)

p1 = sns.heatmap(pt1, ax=ax1, cmap='rainbow', vmax=0.91, vmin=0.7,linewidths=0.05)
ax1.set_title('Performance[0~1]',fontsize=12)
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('max_depth')

#可视化展示随机森林中的第一颗树
# Extract the first tree in the forest
tree_small = forest.estimators_[0]

# Save the tree as a png image
from sklearn.tree import export_graphviz
from pydot import graph_from_dot_file
#export_graphviz(tree_small, out_file = 
#'small_tree.dot', feature_names = feat_labels_important, rounded = True, precision = 1)
 
#(graph, ) = graph_from_dot_file('small_tree.dot')
 
#graph.write_png('small_tree.png');

