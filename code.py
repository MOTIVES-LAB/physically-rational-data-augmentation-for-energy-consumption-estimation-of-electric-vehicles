#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import requests
import os
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import torch
from torch.utils.data import DataLoader,TensorDataset
import joblib
import random
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(mode=False)
from torch.optim import lr_scheduler
import shutil
from matplotlib.patches import Patch
from scipy.stats import gamma
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import AdaBoostRegressor
from scipy.optimize import curve_fit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import boxcox1p
from scipy.special import boxcox
from scipy.stats import boxcox_normmax
from scipy.stats import norm
# 柱状分布图
def norm_comparision_plot(data, figsize=(16, 9), color="#099DD9",
                          ax=None, surround=True, grid=True):
    """
    function: 传入 DataFrame 指定行，绘制其概率分布曲线与正态分布曲线(比较)
    color: 默认为标准天蓝  #F79420:浅橙  ‘green’：直接绿色(透明度自动匹配)
    ggplot 经典三原色：'#F77B72'：浅红, '#7885CB'：浅紫, '#4CB5AB'：浅绿
    ax=None: 默认无需绘制子图的效果；  surround：sns.despine 的经典组合，
                                         默认开启，需要显式关闭
    grid：是否添加网格线，默认开启，需显式关闭                             
    """
#     list1=[4.8,
#     18.6,
#     2.7,
#     11.5,
#     8.5,
#     10.3,
#     14.5,
#     3.7,
#     4.3,
#     16.5,
#     4.5,
#     5.3,
#     7.6]
#     ax1.set_xlabel(data.name)

#     fig, ax1 = plt.subplots(figsize=figsize) # 设置图片大小
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("Kernel Density Estimate")

#     fit=norm: 同等条件下的正态曲线(默认黑色线)；lw-line width 线宽
    # 绘制频数和核密度
#     sns.distplot(data,color=color,hist=False,kde=True,
#                  kde_kws={"color" :color, "lw" :3 }, ax=ax2)
# #     plt.bar(list1,height=0.005,color='#4CB5AB')
    
#     创建第二个y轴并设置标签

#     # 设置第一个y轴的标签和范围
#     ax1.set_ylabel("Frequency")
#     ax1.set_ylim(0, 30)

#     设置第二个y轴的范围
#     ax2.set_ylim(0, 0.5)

    plt.figure(figsize=figsize) # 设置图片大小
#     sns.histplot(data, kde=False,bins=20, ax=ax)

#     fit=norm: 同等条件下的正态曲线(默认黑色线)；lw-line width 线宽
    sns.distplot(data,fit=norm,color=color,hist=True,kde=True,
                 kde_kws={"color" :color, "lw" :3 }, ax=ax)
    (mu, sigma) = norm.fit(data)  # 求同等条件下正态分布的 mu 和 sigma
    # 添加图例：使用格式化输入，loc='best' 表示自动将图例放到最合适的位置
    plt.legend(['Normal dist. ($\mu=$ {:.3f} and $\sigma=$ {:.3f} )'.                format(mu, sigma)] ,loc='upper right')
#     plt.ylabel('Frequency')
#     plt.xticks(fontsize=24)
#     plt.title("Distribution")
    if surround == True:
        # trim=True-隐藏上面跟右边的边框线，left=True-隐藏左边的边框线
        # offset：偏移量，x 轴向下偏移，更加美观
        sns.despine(trim=True, left=True, offset=10)
    if grid == True:
        plt.grid(True)  # 添加网格线
        
def combinations(iterable, r, interval):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(0,r))
    yield tuple(pool[i] for i in indices)
    
    k=0
    
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        if k%interval==0:
            yield tuple(pool[i] for i in indices)
        k += 1


# In[4]:


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100


def smape_error(y_true, y_pred):
    epsilon = 0.1
    summ = np.maximum(0.5 + epsilon, np.abs(y_true) + np.abs(y_pred) + epsilon)
    smape = np.mean(np.abs(y_true - y_pred) / summ) * 2
    return smape


def smape_func(preds, dtrain):
    label = dtrain.get_label().values
    epsilon = 0.1
    summ = np.maximum(0.5 + epsilon, np.abs(label) + np.abs(preds) + epsilon)
    smape = np.mean(np.abs(label - preds) / summ) * 2
    return 'smape', float(smape), False


# In[5]:


def traffic_feature1(data):
    data['Time']=(data['start_time']+data['end_time'])/2
    weekday_condition = data['weekday'].isin([0, 1, 2, 3, 4])
    time_condition_morning = (data['Time'] < 32400) & (data['Time'] > 25200)
    time_condition_evening = (data['Time'] < 64800) & (data['Time'] > 57600)
    time_condition_weekend_morning = (data['Time'] < 11 * 3600) & (data['Time'] > 9 * 3600)
    time_condition_weekend_evening = (data['Time'] < 17 * 3600) & (data['Time'] > 15 * 3600)

    data['Weekdays morning rush hours'] = (weekday_condition & time_condition_morning).astype(int)
    data['Weekdays evening rush hours'] = (weekday_condition & time_condition_evening).astype(int)
    data['Weekdays non-rush hours'] = (weekday_condition & ~time_condition_morning & ~time_condition_evening).astype(int)
    data['Weekends morning rush hours'] = (~weekday_condition & time_condition_weekend_morning).astype(int)
    data['Weekends evening rush hours'] = (~weekday_condition & time_condition_weekend_evening).astype(int)
    data['Weekends non-rush hours'] = (~weekday_condition & ~time_condition_weekend_morning & ~time_condition_weekend_evening).astype(int)
    return data


# In[128]:


# 做差增强

def XGB_paras(df_train,i):
    start=time.time()
    X_train, X_val, Y_train, Y_val = train_test_split(df_train.iloc[:,:i], df_train.iloc[:,i], test_size=0.2,random_state=42)
    # 使用训练集获得超参数
#     trn_data = xgb.DMatrix(X_train, label=Y_train)
#     val_data = xgb.DMatrix(X_val, label=Y_val)
    params = {
        'n_estimators':range(10,500,2),
        'min_child_weight': range(5,101,5),# 最小叶子节点样本数,数值越大,特殊样本的权重越低
    #     'eval_metric': 'mae',
        'max_depth': range(2,7,1),
        'alpha': np.linspace(0.0001,0.2,10),#L1正则化
    #     'lambda': np.linspace(0.0001,0.2,10),#L2正则化
        'colsample_bytree': np.linspace(0.5,0.9,10), # 构建每个树时的子抽样比例。
        'subsample': np.linspace(0.4,0.9,10),# boosting采样比率
        'eta': np.linspace(0.1,0.6,10),# 学习率
    #     'gamma': 0.001,# 在树叶节点进行划分所需要达到的最小损失减少 
    }

    clf=xgb.XGBRegressor(seed=42,nthread=-1,gamma=0.001,booster='dart')

    grid = RandomizedSearchCV(clf,
                              params,
                              scoring='neg_mean_absolute_error',
                              cv = 5,
                              n_iter=500,
                              random_state=42,
                              n_jobs = -1)
    grid.fit(X_train.values,Y_train.values,verbose=100)
    best_estimator = grid.best_estimator_
#     print(best_estimator)
    
#     # 输出最优模型的精度
#     print(grid.best_score_)
    # 模型保存
    best_estimator.save_model("XGboost_train_model.json")
    end=time.time()
    
#     print('XGB Find paras time is %.2f' %(end-start))

    
    score=grid.cv_results_['mean_test_score']
    
    d=pd.DataFrame(score)
    
    d.to_csv(str(i)+'.csv',index=False)
    
    
    
    return best_estimator.get_params()


def RFR_paras(df_train,i):
    # val_data = xgb.DMatrix(X_val, label=Y_val)
    start=time.time()

    params = {
        'n_estimators':range(40,80,1),
        'min_samples_leaf': range(4,50,2),# 最小叶子节点样本数,数值越大,特殊样本的权重越低
    #     'eval_metric': 'mae',
        'max_depth': range(3,7,1),
#         'alpha': np.linspace(0.001,0.2,10),#L1正则化
    #     'lambda': np.linspace(0.0001,0.2,10),#L2正则化
        'min_samples_split': range(30,60,2), # 构建每个树时的子抽样比例。
    #     'gamma': 0.001,# 在树叶节点进行划分所需要达到的最小损失减少 
        
    }
    regr = RandomForestRegressor(random_state=42,max_features='sqrt',criterion='friedman_mse')
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]

    grid = RandomizedSearchCV(regr,
                              params,
                              cv = 5,
                              n_iter=200,
                              random_state=42,
                              n_jobs = -1)
    grid.fit(X_train.values,Y_train.values)
    best_estimator = grid.best_estimator_
    
    print(grid.best_score_)
#     # save
#     joblib.dump(best_estimator, "my_random_forest.joblib")

#     # load
#     loaded_rf = joblib.load("my_random_forest.joblib")
    end=time.time()
    print('RFR find paras time is %.2f' %(end-start))

    return best_estimator.get_params()

def DT_paras(df_train,i):
    # val_data = xgb.DMatrix(X_val, label=Y_val)
    start=time.time()

    params = {
        'min_samples_leaf': range(4,50,2),# 最小叶子节点样本数,数值越大,特殊样本的权重越低
        'max_depth': range(3,12,1),
#         'alpha': np.linspace(0.001,0.2,10),#L1正则化
    #     'lambda': np.linspace(0.0001,0.2,10),#L2正则化
        'min_samples_split': range(30,60,2), # 构建每个树时的子抽样比例。
    #     'gamma': 0.001,# 在树叶节点进行划分所需要达到的最小损失减少 
    }
    
    regr = DecisionTreeRegressor(max_features='sqrt',criterion='friedman_mse')
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]

    grid = RandomizedSearchCV(regr,
                              params,
                              cv = 5,
                              n_iter=200,
                              random_state=42,
                              n_jobs = -1)
    grid.fit(X_train.values,Y_train.values)
    best_estimator = grid.best_estimator_
    
    print(grid.best_score_)
#     # save
#     joblib.dump(best_estimator, "my_random_forest.joblib")

#     # load
#     loaded_rf = joblib.load("my_random_forest.joblib")
    end=time.time()
    print('DT find paras time is %.2f' %(end-start))

    return best_estimator.get_params()


def KNN_paras(df_train,i):
    # val_data = xgb.DMatrix(X_val, label=Y_val)
    start=time.time()

    params = {
        'n_neighbors': range(5,100,5),# 最小叶子节点样本数,数值越大,特殊样本的权重越低
        'weights' :['uniform', 'distance']
    }
    
    regr = KNeighborsRegressor()
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]

    grid = RandomizedSearchCV(regr,
                              params,
                              cv = 5,
                              n_iter=200,
                              random_state=42,
                              n_jobs = -1)
    grid.fit(X_train.values,Y_train.values)
    best_estimator = grid.best_estimator_
    
    print(grid.best_score_)
#     # save
#     joblib.dump(best_estimator, "my_random_forest.joblib")

#     # load
#     loaded_rf = joblib.load("my_random_forest.joblib")
    end=time.time()
    print('KNN find paras time is %.2f' %(end-start))

    return best_estimator.get_params()

def Ada_paras(df_train,i):
    start=time.time()
    X_train, X_val, Y_train, Y_val = train_test_split(df_train.iloc[:,:i], df_train.iloc[:,i], test_size=0.2,random_state=42)
    # 使用训练集获得超参数
#     trn_data = xgb.DMatrix(X_train, label=Y_train)
#     val_data = xgb.DMatrix(X_val, label=Y_val)
    params = {
        'n_estimators':range(50,200,2),
        'learning_rate': np.linspace(0.005,0.1,25),# 学习率
    #     'gamma': 0.001,# 在树叶节点进行划分所需要达到的最小损失减少 
    }

    clf=AdaBoostRegressor()

    grid = RandomizedSearchCV(clf,
                              params,
                              cv = 5,
                              n_iter=200,
                              random_state=42,
                              n_jobs = -1)
    grid.fit(X_train.values,Y_train.values)
    best_estimator = grid.best_estimator_
    print(best_estimator)
    
    # 输出最优模型的精度
    print(grid.best_score_)
    # 模型保存
    end=time.time()
    
    print('Ada Find paras time is %.2f' %(end-start))
    
    print(grid.grid_scores_)

    return best_estimator.get_params()


def LGB_paras(df_train,i):
    start=time.time()
    X_train, X_val, Y_train, Y_val = train_test_split(df_train.iloc[:,:i], df_train.iloc[:,i], test_size=0.2,random_state=42)
    # 使用训练集获得超参数
#     trn_data = xgb.DMatrix(X_train, label=Y_train)
#     val_data = xgb.DMatrix(X_val, label=Y_val)
    params = {
        'num_leaves':range(30,80,5),
        'max_depth':range(3,10,1),
        'n_estimators':range(50,200,10),
        'learning_rate': np.linspace(0.005,0.15,10),# 学习率
    #     'gamma': 0.001,# 在树叶节点进行划分所需要达到的最小损失减少 
    }

    clf=lgb.LGBMRegressor()

    grid = RandomizedSearchCV(clf,
                              params,
                              cv = 5,
                              n_iter=200,
                              scoring='neg_mean_absolute_error',
                              random_state=888,
                              n_jobs = -1)
    grid.fit(X_train.values,Y_train.values)
    best_estimator = grid.best_estimator_
    
#     print(best_estimator)
#     # 输出最优模型的精度
#     print(grid.best_score_)
    # 模型保存
    end=time.time()
    
#     print('lgb Find paras time is %.2f' %(end-start))
    
    score=grid.cv_results_['mean_test_score']
    
    d=pd.DataFrame(score)
    
    d.to_csv(str(i)+'.csv',index=False)
    
    return best_estimator.get_params()



def DT_model(df_train,df_val,DT_para,i):
    dt=DecisionTreeRegressor(**DT_para)
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]
    dt.fit(X_train.values,Y_train.values)
    X_val=df_val.iloc[:,:i]
    y_val=dt.predict(X_val.values)
    y_train=dt.predict(X_train.values)
    return y_train,y_val

def RFR_model(df_train,df_val,RFR_para,i):
    dt=RandomForestRegressor(**RFR_para)
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]
    dt.fit(X_train.values,Y_train.values)
    X_val=df_val.iloc[:,:i]
    y_val=dt.predict(X_val.values)
    y_train=dt.predict(X_train.values)
    return y_train,y_val

def LGB_model(df_train,df_val,RFR_para,i):
    LGB = lgb.LGBMRegressor(**RFR_para)
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]
    LGB.fit(X_train.values,Y_train.values)
    X_val=df_val.iloc[:,:i]
    y_val=LGB.predict(X_val.values)
    y_train=LGB.predict(X_train.values)
    return y_train,y_val


def KNN_model(df_train,df_val,KNN_para,i):
    knr=KNeighborsRegressor(**KNN_para)
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]
    knr.fit(X_train.values,Y_train.values)
    X_val=df_val.iloc[:,:i]
    y_val=knr.predict(X_val.values)
    y_train=knr.predict(X_train.values)
    return y_train,y_val


def XGB_model(df_train,df_val,XGB_para,i):

    clf=xgb.XGBRegressor()
    clf.set_params(**XGB_para)
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]
    clf.fit(X_train.values,Y_train.values)
    X_val=df_val.iloc[:,:i]
    y_val=clf.predict(X_val)
    y_train=clf.predict(X_train)
    return y_train,y_val

def linear_model(df_train,df_val,i):
    
    clf=LinearRegression()
    
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]
    clf.fit(X_train.values,Y_train.values)
    X_val=df_val.iloc[:,:i]
    Y_val=df_val.iloc[:,i]
    y_val=clf.predict(X_val.values)
    y_train=clf.predict(X_train.values)
    yv=y_val
    yt=y_train
    
    return yt,yv

def Ada_model(df_train,df_val,Ada_para,i):
    clf=AdaBoostRegressor(**Ada_para)
    X_train=df_train.iloc[:,:i]
    Y_train=df_train.iloc[:,i]
    clf.fit(X_train.values,Y_train.values)
    X_val=df_val.iloc[:,:i]
    y_val=clf.predict(X_val.values)
    y_train=clf.predict(X_train.values)
    return y_train,y_val


# In[5]:



def norm_comparision_plot(data_list, labels, figsize=(16, 9), color_list=["#099DD9", "#F79420", "green"],
                          ax=None, surround=True, grid=True):
    """
    function: 传入多个数据集，绘制它们的概率分布曲线与正态分布曲线进行比较
    data_list: 数据集列表
    labels: 数据集标签列表，用于图例
    figsize: 图片大小，默认为 (16, 9)
    color_list: 数据集颜色列表，默认为 ["#099DD9", "#F79420", "green"]
    ax: 绘图的坐标轴对象，默认为 None，即创建一个新的图
    surround: 是否进行 sns.despine 的经典组合，默认为 True，需要显式关闭
    grid: 是否添加网格线，默认为 True，需显式关闭
    """

    if ax is None:
        plt.figure(figsize=figsize)
# fit=beta, fit_kws={"color": color_list[i],"linestyle": '--'},
    for i, data in enumerate(data_list):
        sns_plot =sns.distplot(data,  color=color_list[i], hist=True, kde=True,
                               kde_kws={"color": color_list[i], "lw": 3}, ax=ax)
#         lines = sns_plot.get_lines()
#         lines[1].set_color(color_list[i])
    legends = [Patch(facecolor=color_list[i], label=label) for i, label in enumerate(labels)]
    plt.legend(handles=legends, loc='upper right')

#     legends = ['Normal dist. ($\mu={:.3f}$, $\sigma={:.3f}$)'.format(np.mean(data), np.std(data)) 
#                for data in data_list]
#     plt.legend(legends, labels, loc='upper right')

    if surround:
        sns.despine(trim=True, left=True, offset=10)
    if grid:
        plt.grid(True)
    


# # 数据增强过程

# ## 特征增强

# In[1]:


# 特征增强

def baseline_1(readtrain_path,readval_path,savefile_path):

    list1=os.listdir(readtrain_path)
    namelist=['grad'+str(i) for i in range(13)]+['Time','Weekdays morning rush hours','Weekdays evening rush hours','Weekdays non-rush hours','Weekends morning rush hours','Weekends evening rush hours','Weekends non-rush hours']

    Dnamelist=['Dgrad'+str(i) for i in range(13)]+['DTime','DWeekdays morning rush hours','DWeekdays evening rush hours','DWeekdays non-rush hours','DWeekends morning rush hours','DWeekends evening rush hours','DWeekends non-rush hours']
    name=['num',"drive_time","mileage","soc_start",'Amb_temp','grade_cv','grade_var','bat_min',"bat_max",
                              "elevation","weekday","start_time",'spd_std','a_std','a_mean','end_time',"spdmean",'tech_speed',"acc_mean","acc_95","dec_5","dec_mean",
                              "stop_time","pedal_dev","steerspd_dev", "steerang_dev","heat_rate","evap_rate",
                              'Energy consumption','ECR','I_spd','I_grav','I_brkh_test','I_brkl_test','I_brk_test','I_slw']+namelist
    Dname=['Dnum',"Ddrive_time","Dmileage","Dsoc_start",'DAmb_temp','Dgrade_cv','Dgrade_var','Dbat_min',"Dbat_max",
                              "Delevation","Dweekday","Dstart_time",'Dspd_std','Da_std','Da_mean','Dend_time',"Dspdmean",'Dtech_speed',"Dacc_mean","Dacc_95","Ddec_5","Ddec_mean",
                              "Dstop_time","Dpedal_dev","Dsteerspd_dev", "Dsteerang_dev","Dheat_rate","Devap_rate",
                              'DEnergy consumption','DECR','DI_spd','DI_grav','DI_brkh_test','DI_brkl_test','DI_brk_test','DI_slw']+Dnamelist

    list2=os.listdir(readval_path)

    for d in range(len(list1)):
        data = pd.read_csv(readtrain_path+"\\"+list1[d])

        data['Time']=(data['start_time']+data['end_time'])/2
        weekday_condition = data['weekday'].isin([0, 1, 2, 3, 4])
        time_condition_morning = (data['Time'] < 32400) & (data['Time'] > 25200)
        time_condition_evening = (data['Time'] < 64800) & (data['Time'] > 57600)
        time_condition_weekend_morning = (data['Time'] < 11 * 3600) & (data['Time'] > 9 * 3600)
        time_condition_weekend_evening = (data['Time'] < 17 * 3600) & (data['Time'] > 15 * 3600)

        data['Weekdays morning rush hours'] = (weekday_condition & time_condition_morning).astype(int)
        data['Weekdays evening rush hours'] = (weekday_condition & time_condition_evening).astype(int)
        data['Weekdays non-rush hours'] = (weekday_condition & ~time_condition_morning & ~time_condition_evening).astype(int)
        data['Weekends morning rush hours'] = (~weekday_condition & time_condition_weekend_morning).astype(int)
        data['Weekends evening rush hours'] = (~weekday_condition & time_condition_weekend_evening).astype(int)
        data['Weekends non-rush hours'] = (~weekday_condition & ~time_condition_weekend_morning & ~time_condition_weekend_evening).astype(int)


        data0 = pd.read_csv(readval_path+"\\"+list2[d])

        data0['Time']=(data0['start_time']+data0['end_time'])/2
        data0['Weekdays morning rush hours'] = (weekday_condition & time_condition_morning).astype(int)
        data0['Weekdays evening rush hours'] = (weekday_condition & time_condition_evening).astype(int)
        data0['Weekdays non-rush hours'] = (weekday_condition & ~time_condition_morning & ~time_condition_evening).astype(int)
        data0['Weekends morning rush hours'] = (~weekday_condition & time_condition_weekend_morning).astype(int)
        data0['Weekends evening rush hours'] = (~weekday_condition & time_condition_weekend_evening).astype(int)
        data0['Weekends non-rush hours'] = (~weekday_condition & ~time_condition_weekend_morning & ~time_condition_weekend_evening).astype(int)



        data1=data[name]

        data2=data0[name]

        data1['Time']=(data1['start_time']+data1['start_time'])/2
        weekday_condition = data1['weekday'].isin([0, 1, 2, 3, 4])
        time_condition_morning = (data1['Time'] < 32400) & (data1['Time'] > 25200)
        time_condition_evening = (data1['Time'] < 64800) & (data1['Time'] > 57600)
        time_condition_weekend_morning = (data1['Time'] < 11 * 3600) & (data1['Time'] > 9 * 3600)
        time_condition_weekend_evening = (data1['Time'] < 17 * 3600) & (data1['Time'] > 15 * 3600)

        data1['Weekdays morning rush hours'] = (weekday_condition & time_condition_morning).astype(int)
        data1['Weekdays evening rush hours'] = (weekday_condition & time_condition_evening).astype(int)
        data1['Weekdays non-rush hours'] = (weekday_condition & ~time_condition_morning & ~time_condition_evening).astype(int)
        data1['Weekends morning rush hours'] = (~weekday_condition & time_condition_weekend_morning).astype(int)
        data1['Weekends evening rush hours'] = (~weekday_condition & time_condition_weekend_evening).astype(int)
        data1['Weekends non-rush hours'] = (~weekday_condition & ~time_condition_weekend_morning & ~time_condition_weekend_evening).astype(int)

        data2['Time']=(data2['start_time']+data2['start_time'])/2
        weekday_condition = data2['weekday'].isin([0, 1, 2, 3, 4])
        time_condition_morning = (data2['Time'] < 32400) & (data2['Time'] > 25200)
        time_condition_evening = (data2['Time'] < 64800) & (data2['Time'] > 57600)
        time_condition_weekend_morning = (data2['Time'] < 11 * 3600) & (data2['Time'] > 9 * 3600)
        time_condition_weekend_evening = (data2['Time'] < 17 * 3600) & (data2['Time'] > 15 * 3600)

        data2['Weekdays morning rush hours'] = (weekday_condition & time_condition_morning).astype(int)
        data2['Weekdays evening rush hours'] = (weekday_condition & time_condition_evening).astype(int)
        data2['Weekdays non-rush hours'] = (weekday_condition & ~time_condition_morning & ~time_condition_evening).astype(int)
        data2['Weekends morning rush hours'] = (~weekday_condition & time_condition_weekend_morning).astype(int)
        data2['Weekends evening rush hours'] = (~weekday_condition & time_condition_weekend_evening).astype(int)
        data2['Weekends non-rush hours'] = (~weekday_condition & ~time_condition_weekend_morning & ~time_condition_weekend_evening).astype(int)

        columns = Dname + data1.columns.tolist()

        data = data1.values

        # 计算每一行数据与其他行数据的差
        diff_data = data[:, np.newaxis, :] - data

        # 将差数据和原始数据按列拼接
        result_data = np.concatenate([diff_data.reshape(-1, data.shape[1]), data.repeat(data.shape[0], axis=0)], axis=1)
        # 转换为DataFrame
        result_df = pd.DataFrame(result_data, columns=columns)
        # 打印结果DataFrame

        p = data1['ECR'].values  # 获取要重复的列的数据
        repeated_data = np.tile(p, len(data1))  # 重复数据十次
        result_df['EC']=repeated_data

        result_df=result_df[(result_df['Ddrive_time']!=0)&(result_df['Dgrade_cv']!=0)]

        data_t = data2.values

        diff_data_test = data_t[:, np.newaxis, :] - data


        # 将差数据和原始数据按列拼接
        test_datas = np.concatenate([diff_data_test.reshape(-1, data.shape[1]), data_t.repeat(data.shape[0], axis=0)], axis=1)

        test_datas = pd.DataFrame(test_datas, columns=columns)    

        p = data1['ECR'].values  # 获取要重复的列的数据
        repeated_data = np.tile(p, len(data2))  # 重复数据十次
        test_datas['EC']=repeated_data

        result_df.to_csv(savefile_path+'\\'+list2[d],index=False)

        test_datas.to_csv(savefile_path+'\\'+list1[d],index=False)
        
        
        return 
    


# ## 物理增强

# In[ ]:


# 按行程为单位，带五折交叉验证

def routes(origin,destination):
    key = '0bb8d13cf68d1b6259d0efda6d007456' ##输入自己key
    parameters = {'key':key,'origin':origin,'destination':destination,'extensions':'all','strategy':0}
    ##参数的输入，可以按照自己的需求选择出行时间最短，出行距离最短，不走高速等方案，结合自己需求设置，参考手册
    response = requests.get('https://restapi.amap.com/v3/direction/driving?parameters',params=parameters)
    text = json.loads(response.text)
#     duration = text['route']['paths'][0]['duration'] ##出行时间
    ## 可以自己打印text看一下，能提取很多参数，出行时间、出行费用、出行花费等看自己需求提取
    return text



# 特征函数
def flager(df):
    
    data_progress=df
    d=data_progress[data_progress['spd']==0]
    d=d.reset_index()

    index=[0]
    j=0
    k=0
    for i in range(len(d.loc[d['index'].diff()!=1,'index'])):

        # 停车末点-停车初点
        index_start=d.loc[d['index'].diff()!=1,'index'].values[i]
        index_end=d.loc[d['index'].diff(-1)!=-1,'index'].values[i]

        index.append(int(index_start+(index_end-index_start)/2))
        if index_end-index_start>60 and sum(data_progress.loc[index[j]:index[i+1],'spd']/3.6)*0.05>1:
#             print(sum(data_progress.loc[index[j]:index[i+1],'spd']/3.6)*0.05,index[j],index[i+1],k)
            data_progress.loc[index[j]:index[i+1],'flag_1']=k
            k+=1
            j=i+1
        else:
            pass        
    if len(d.loc[d['index'].diff()!=1,'index'])==0 or 'flag_1' not in data_progress.columns:
        data_progress['flag_1']=0
        data_progress.to_csv('kkzmhs.csv')
        print('meifenduan')
    return data_progress[data_progress['flag_1'].notnull()]
#         print(index_end-index_start,sum(data_progress.loc[index[j]:index[i+1],'spd'])/3.6*0.05>500)
#         print(i,j)
#     if i+1==len(d.loc[d['index'].diff()!=1,'index']):
#         data_progress.loc[index[i+1]:,'flag_1']=i


def I_slw(spd):
    v_bar = (np.array(spd[0:len(spd)-1])+np.array(spd[1:len(spd)]))/2
    i_slw=1/sum((v_bar))*len(spd)
#     I_slw=1/sum(spd)*len(spd)
    return i_slw
def func_v_bar(data_v):
    v_bar = (np.array(data_v[0:len(data_v)-1])+np.array(data_v[1:len(data_v)]))/2
    return v_bar
def func_v_bar0(data_v):
    v_bar0 = np.array(data_v[0:len(data_v)-1])
    return v_bar0
def func_v_bar1(data_v):
    v_bar1 = np.array(data_v[1:len(data_v)])
    return v_bar1
######
def integal_trapezoidal(data,ts):
    e_integal = sum((np.array(data[0:len(data)])+np.array(data[1:len(data)+1]))/2*ts)
    return e_integal
# def func_I_brk(v,ts,para1,para2,h):   ###制动强度   ##para1[质量] para2[速度敏感度，重力加速度，效率，dt，基值]
def func_I_brk(v,ts,para1,para2):
    v = np.array(v)/3.6
    v_bar = func_v_bar(v)
    v_bar0= func_v_bar0(v)
    v_bar1= func_v_bar1(v)
#     dh = (-np.array(h[0:len(h)-1])+np.array(h[1:len(h)]))
    numb_v = len(v)-1
    E_brk = []
    m = para1
    beta0 = para2[4]
    beta_spd = para2[0]
    g = para2[1]
    yita =para2[2]
    dt = para2[3]
    E_brksh = []
    E_brksl = []
    for i in range(numb_v):
        E_brki = m*((v_bar0[i])**2-(v_bar1[i])**2)/2-para_a*ts*(v_bar0[i]+v_bar1[i])/2-para_b*ts*((v_bar0[i]*3.6)**2+(v_bar1[i]*3.6)**2+(v_bar0[i]*3.6)*(v_bar1[i]*3.6))/3-para_c*ts*(1/4)*((v_bar0[i]*3.6)**3+(v_bar0[i]*3.6)**2*v_bar1[i]*3.6+(v_bar0[i]*3.6)*(v_bar1[i]*3.6)**2+(v_bar1[i]*3.6)**3)
#         E_brki = m*((v_bar0[i])**2-(v_bar1[i])**2)/2
        if E_brki<0:
            E_brki = 0
        if v_bar[i]*3.6>20:
            E_brksh.append(E_brki)
        else:
            E_brksl.append(E_brki)
    I_brkh_test=20*(sum(E_brksh))/sum(v_bar)
    I_brkl_test=20*(sum(E_brksl))/sum(v_bar)
    I_brk_test=20*(sum(E_brksh)+sum(E_brksl))/sum(v_bar)
    return I_brkh_test,I_brkl_test,I_brk_test

def grade_std(df):
    df=df[['spd','elevation']]
    df1 = df.loc[ : :10, : ].reset_index(drop=True)
    df2 = df.loc[1 : :10, : ].reset_index(drop=True)
    df3 = df.loc[2 : :10, : ].reset_index(drop=True)
    df4 = df.loc[3 : :10, : ].reset_index(drop=True)
    df5 = df.loc[4 : :10, : ].reset_index(drop=True)
    df6 = df.loc[5 : :10, : ].reset_index(drop=True)
    df7 = df.loc[6 : :10, : ].reset_index(drop=True)
    df8 = df.loc[7 : :10, : ].reset_index(drop=True)
    df9 = df.loc[8 : :10, : ].reset_index(drop=True)
    df10= df.loc[9 : :10, : ].reset_index(drop=True)
    df_use=(df1+df2+df3+df4+df5+df6+df7+df8+df9+df10)/10
    del df1,df2,df3,df4,df5,df6,df7,df8,df9,df10

    #     海拔差值
    df_use['ele_diff']=df_use['elevation'].diff()
    df_use.loc[0,'ele_diff']=0
    df_use.loc[abs(df_use['ele_diff'])>2,'ele_diff']=0

    #     c['panduan']为上下坡的判断依据
    c=df_use.loc[df_use['ele_diff']!=0,'ele_diff'].reset_index()
    c['next_diff']=c['ele_diff'].shift(-1)
    # c.loc[0,'next_diff']=0
    c['panduan']=c['next_diff']*c['ele_diff']


    start=0
    frq=0.5
    df_use['grade']=''

    #     认为上坡的坡度是相同的

    for i in c.loc[c['panduan']<0,'index']:
        h=df_use.loc[i,'elevation']-df_use.loc[start,'elevation']
        x=(sum(df_use.loc[start:i,'spd'])*frq)/3.6
        if x !=0:
            df_use.loc[start:i,'grade']=h/x
        else:
            df_use.loc[start:i,'grade']=0
        start=i

    h=df_use.loc[len(df_use)-2,'elevation']-df_use.loc[start,'elevation']
    x=(sum(df_use.loc[start:len(df_use)-2,'spd'])*frq)/3.6
    if x !=0:
        df_use.loc[start:len(df_use),'grade']=h/x
    else:
        df_use.loc[start:len(df_use),'grade']=0
    df_use=df_use[df_use['grade'].notnull()]
    
    if df_use['grade'].mean()!=0:
        cv=df_use['grade'].std()/df_use['grade'].mean()
    else:
        cv=0
    return cv,df_use['grade'].std()

def feature(df,num):
    test_data =df
    test_data['spd']=abs(test_data['spd'])
#     test_data['spd']=test_data['spd']*3.6
    times=len(test_data)*0.05
    mileage=sum(test_data['spd']/3.6*0.05)/1000
    soc_start=test_data.loc[0,'SOC']
    temp=test_data['tempAmbient'].mean()
    bat_min=test_data['Pack_min'].mean()
    bat_max=test_data['Pack_max'].mean()
    elevation=test_data['elevation'].values[-1]-test_data['elevation'].values[0]
    Total_Disc=test_data['Total_Disc'].values[-1]-test_data['Total_Disc'].values[0]
    Total_c=test_data['Total_c'].values[0]-test_data['Total_c'].values[-1]
    ec=sum(test_data['vol']*test_data['cur'])/1000/3600*0.05
#     date='-'.join(test_data.loc[0,'timestamp'].split('-')[0:3])
#     start_time=test_data.loc[0,'timestamp'].split('-')[3:]
#     start_time[0]=int(start_time[0])+8
#     start_time[1]=int(start_time[1])+46
#     if start_time[1]>60:
#         start_time[0]+=1
#         start_time[1]=start_time[1]-60
#     start_time=str(start_time[0])+'-'+str(start_time[1])+'-'+str(start_time[2])

    """_________________________________________________________________________________________"""
    test_data['miao']=test_data['Time'].apply(lambda x: (datetime.strptime(x.split('.')[0],"%Y-%m-%d %H:%M:%S")-datetime.strptime(x.split(' ')[0],"%Y-%m-%d")).total_seconds())

    t_idle=len(test_data[test_data['spd']==0])*0.05
        
    s_long = test_data['long'].values[0]
    s_lat = test_data['lat'].values[0]
    e_lat = test_data['lat'].values[-1]
    e_long = test_data['long'].values[-1]
    origin=str(s_long)+','+str(s_lat)
    destination=str(e_long)+','+str(e_lat)
#     text=routes(origin,destination)
#     lights_number=text['route']['paths'][0]['traffic_lights']
    date=test_data['Time'].values[0].split(' ')[0]
    
    spdmean = test_data['spd'].mean()
    test_data1=test_data[0::20].reset_index(drop=True)
    test_data1['spd_1']=test_data1['spd'].shift(1)
    test_data1.loc[0,'spd_1']=test_data1.loc[1,'spd_1']
    test_data1['a']=(test_data1['spd']-test_data1['spd_1'])/3.6
    acc_90=test_data1.loc[test_data1['a']>0,'a'].quantile(0.9)
    acc_mean=test_data1.loc[test_data1['a']>=acc_90,'a'].mean()
    dec_10=test_data1.loc[test_data1['a']<0,'a'].quantile(0.1)
    dec_mean=test_data1.loc[test_data1['a']<=dec_10,'a'].mean()
    acc_95=test_data1['a'].quantile(0.95)
    dec_5=test_data1['a'].quantile(0.05)
    tech_speed=test_data.loc[test_data['spd']!=0,'spd'].mean()
    
    test_data2=test_data[0::1200].reset_index(drop=True)
    test_data2['x_distance']=test_data2['Odometer'].diff()
    test_data2['h_distance']=test_data2['elevation'].diff()
    test_data2.loc[0,'x_distance']=0.00001
    test_data2.loc[0,'h_distance']=0
    test_data2.loc[test_data2['x_distance']==0,'x_distance']=0.00001
    test_data2['gradient']=test_data2['h_distance']/test_data2['x_distance']/1000
    bins = [-float('inf'), -0.11,-0.09,-0.07,-0.05, -0.03, -0.01, 0.01, 0.03, 0.05,0.07, 0.09, 0.11, float('inf')]
    labels=[i for i in range(13)]
    test_data2['category'] = pd.cut(test_data2['gradient'], bins=bins, labels=labels)
    gradient_list=[0 for i in range(13)]
    for i in test_data2['category'].unique():
        gradient_list[i]=sum(test_data2.loc[test_data2['category']==i,'x_distance'])/sum(test_data2['x_distance'])
    
    x=[0]+test_data[test_data['spd']<0.1][0:-1].index.tolist()
    y=test_data[test_data['spd']<0.1].index.tolist()
    if len(x)==len(y):
        data_stop=pd.DataFrame({'a':x,'b':y})
        stop_time=data_stop[(data_stop['b']-data_stop['a']) != 1].shape[0]+1
    else:
        stop_time=0
    pedal=np.percentile(test_data['pedal_pos'],75)-np.percentile(test_data['pedal_pos'],25)
    steerspd=np.percentile(test_data['SteeringSpeed129'],75)-np.percentile(test_data['SteeringSpeed129'],25)
    steerang=np.percentile(test_data['SteeringAngle129'],75)-np.percentile(test_data['SteeringAngle129'],25)
    heat_rate=sum(test_data['lheat']+test_data['rheat'])/2/len(test_data)
    evap_rate=sum(test_data['Evap_enable'])/len(test_data)
    start_time=test_data['miao'].values[0]
    end_time=test_data['miao'].values[-1]
    weekday=time.strptime(test_data.loc[0,'Time'],"%Y-%m-%d %H:%M:%S.%f").tm_wday
#     Congestion=data_cong.loc[start_time:end_time,'cong_index'].mean()
    a_mean=test_data1['a'].mean()
    spd_std=test_data['spd'].std()
    a_std=test_data1['a'].std()
#     拥堵指数
    grade_cv,grade_var=grade_std(test_data)
    ecr=ec/mileage*1000
    I_grav=elevation/mileage*1919/1000
    I_brkh_test,I_brkl_test,I_brk_test=func_I_brk(test_data['spd'],0.05,para1,para2)
    i_spd=Ispd(test_data['spd'])
    i_slw=I_slw(test_data['spd'])

    data_result=pd.DataFrame([num,times,mileage,soc_start,temp,grade_cv,grade_var,bat_min,bat_max,elevation,weekday,start_time,spd_std,a_std,a_mean,
                              end_time,spdmean,tech_speed,acc_mean,acc_95,dec_5,dec_mean,stop_time,pedal,steerspd,steerang,heat_rate,evap_rate,
                              origin,destination,ec,ecr,i_spd,I_grav,I_brkh_test,I_brkl_test,I_brk_test,i_slw,date]+gradient_list).T
    if np.any(data_result.isnull()):
        test_data.to_csv('kkzmhs.csv')
        print('kong')
    else:
        pass
    
    return data_result

def main_h(df,d,name):
    path6=r'D:\Tesla model 3\tesla_flag\物理增强_无时间限制'
    path7=path6+'\\'+name
    isExists=os.path.exists(path7) #判断路径是否存在，存在则返回true
    if not isExists:
        os.makedirs(path7)
    b=1  # 行程片段编号
    index_df=feature(df,0)
    data=df
    data=data.reset_index(drop=True)
#     数据处理与准备工作
    data=data_deal(data)
    data=flager(data)
    flag=list(data['flag_1'].unique())
    print(f'flag数量为:{len(flag)}')

    data_name=[]
    s=1

    #     segment 切分

    for a in range(len(flag)):
        mileage_len=sum(data.loc[data['flag_1']==a,'spd']/3.6*0.05)
        if mileage_len<500:
            data.loc[data['flag_1']==a,'flag_1']=a+1
    flag=list(data['flag_1'].unique())
    print(f'新flag数量为:{len(flag)}')
    
    for a in range(len(flag)):
        data_name.append('data'+str(s))
        data_name[s-1]=data[data['flag_1']==flag[a]]
        s+=1

    #     segment 重组
        

    for i in range(len(data_name)):
        for j in range(i,len(data_name)):
            data_out=pd.concat(data_name[i:j+1], ignore_index=True).reset_index(drop=True)
            if len(data_out[data_out['spd']==0])/len(data_out)<0.6:
                dataframe=feature(data_out,b)
                index_df=pd.concat([index_df,dataframe])
                b+=1

    if len(index_df)>0:
        index_df.columns=['num',"drive_time","mileage","soc_start",'Amb_temp','grade_cv','grade_var','bat_min',"bat_max",
                          "elevation","weekday","start_time",'spd_std','a_std','a_mean','end_time',"spdmean",'tech_speed',"acc_mean","acc_95","dec_5","dec_mean",
                          "stop_time","pedal_dev","steerspd_dev", "steerang_dev","heat_rate","evap_rate",'origin','destination',
                          'Energy consumption','ECR','I_spd','I_grav','I_brkh_test','I_brkl_test','I_brk_test','I_slw','date']+namelist
        index_df.to_csv(path7+'\\'+str(d)+'_'+'feature.csv',index=False)
    else:
        df.to_csv(str(b)+'data.csv')
def data_deal(data):
    try:
        start=data[data['spd']>15].index[0]
        if data.loc[0,'spd']!=0:
            true_start=data[data['spd']==0].index[0]
            data=data[true_start:].reset_index(drop=True)
            true_start_1=data[data['spd']>0].index[0]
            if true_start_1-500>0:
                data=data[true_start_1-500:]
            else:
                data=data[true_start_1:]
        else:
            data_0=data[:start]
            start_=data_0[data_0['spd']==0].index[-1]
            if start_-500>0:
                data=data[start_-500:]
            else:
                data=data[data_0[data_0['spd']==0].index[0]:]

        data=data.reset_index(drop=True)

        end=len(data)-data[data['spd']!=0].index[-1]
        if end <10 :
            true_end=data[data['spd']==0].index[-1]
            data=data[:true_end]
        elif end >500:
            data=data[:len(data)+500-end]
        else:
            pass
        data=data.reset_index(drop=True)

    #     处理经纬度
        data.loc[data['elevation']>300,'elevation']=None
        data['elevation'] = data['elevation'].fillna(method='ffill')
        data['elevation'] = data['elevation'].fillna(method='bfill')

        if len(data[data['elevation'].diff()>4])>0:
            print(f"经纬度与海拔采样时间相差：{len(data.loc[data[data['elevation'].diff()>4].index[0]:,:])}")
            if len(data.loc[data[data['elevation'].diff()>4].index[0]:,:])<400:
                data.loc[data[data['elevation'].diff()>4].index[0]:,'elevation']=None
            elif len(data.loc[:data[data['elevation'].diff()>4].index[0],:])<400:
                data.loc[:data[data['elevation'].diff()>4].index[0]-1,'elevation']=None
            else :
                data.to_csv('kkzmhs.csv')
                print('kkzmhs')
                d0=data.loc[:data[data['elevation'].diff()>4].index[0]-1]
                d1=data.loc[data[data['elevation'].diff()>4].index[0]:]
                if len(d0)>len(d1):
                    data=d0
                else:
                    data=d1
            data['elevation'] = data['elevation'].fillna(method='ffill')
            data['elevation'] = data['elevation'].fillna(method='bfill')
            data=data.reset_index(drop=True)
            print(abs(data['elevation'].diff()).max())
    except:
        data.to_csv('kkzmhs.csv')
        print('zmhsn')
        data=pd.DataFrame()
    return data

def func_v_bar(data_v):
    v_bar = (np.array(data_v[0:len(data_v)-1])+np.array(data_v[1:len(data_v)]))/2
    return v_bar
def Ispd(spd):
    v_bar = (np.array(spd[0:len(spd)-1])+np.array(spd[1:len(spd)]))/2
    if len(spd)>1:
        v_bar = func_v_bar(spd)/3.6
        I_spd = sum(v_bar**3)/sum((v_bar))
    elif len(spd)==1:
        I_spd = np.array(spd)
    return I_spd




def proposed(travelpath,trainsavepath,valsavepath):

    list1=os.listdir(travelpath)
    list1.sort(key=lambda x:int(x[:-4]))
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(list1)):
        df_input=pd.read_csv(travelpath+'\\'+list1[trn_idx[0]])
        for d in range(1,len(trn_idx)-1):
            df1=pd.read_csv(travelpath+'\\'+list1[trn_idx[d+1]])
            if df_input['SOC'].values[-1]-df1['SOC'].values[0]<0.2 and df1['elevation'].values[0]==df_input['elevation'].values[-1] and abs(df1['Odometer'].values[0]-df_input['Odometer'].values[-1])<0.2:
                df_input=pd.concat([df_input,df1], ignore_index=True)
            else:
                main_h(df_input,d,str('train')+str(fold_))
                df_input=df1
        print(val_idx,fold_)


        b=0
        index_df=pd.DataFrame()
        for d in trn_idx:
            data=pd.read_csv(travelpath+'\\'+list1[d])
        #     数据处理与准备工作
            data=data_deal(data)
            if len(data)>0:
                dataframe=feature(data,b)
                index_df=pd.concat([index_df,dataframe])
                b+=1
        index_df.columns=['num',"drive_time","mileage","soc_start",'Amb_temp','grade_cv','grade_var','bat_min',"bat_max",
                              "elevation","weekday","start_time",'spd_std','a_std','a_mean','end_time',"spdmean",'tech_speed',"acc_mean","acc_95","dec_5","dec_mean",
                              "stop_time","pedal_dev","steerspd_dev", "steerang_dev","heat_rate","evap_rate",'origin','destination',
                              'Energy consumption','ECR','I_spd','I_grav','I_brkh_test','I_brkl_test','I_brk_test','I_slw','date']+namelist
        index_df.to_csv(trainsavepath+'\\'+str(fold_)+'_train.csv',index=False)

        b=0
        index_df=pd.DataFrame()
        for d in val_idx:
            data=pd.read_csv(travelpath+'\\'+list1[d])
        #     数据处理与准备工作
            data=data_deal(data)
            if len(data)>0:
                dataframe=feature(data,b)
                index_df=pd.concat([index_df,dataframe])
                b+=1
        index_df.columns=['num',"drive_time","mileage","soc_start",'Amb_temp','grade_cv','grade_var','bat_min',"bat_max",
                              "elevation","weekday","start_time",'spd_std','a_std','a_mean','end_time',"spdmean",'tech_speed',"acc_mean","acc_95","dec_5","dec_mean",
                              "stop_time","pedal_dev","steerspd_dev", "steerang_dev","heat_rate","evap_rate",'origin','destination',
                              'Energy consumption','ECR','I_spd','I_grav','I_brkh_test','I_brkl_test','I_brk_test','I_slw','date']+namelist
        index_df.to_csv(valsavepath+'\\'+str(fold_)+'_val.csv',index=False)





# ## 传统增强

# In[11]:


# random combination

def random_select_files(input_folder, output_folder_train, output_folder_val, percentage=20, random_seed=42):
    # 设置随机种子
    random.seed(random_seed)

    # 获取文件夹中所有文件的列表
    all_files = os.listdir(input_folder)

    # 计算需要选择的文件数量
    num_files_to_select = int(len(all_files) * percentage / 100)

    # 随机选择文件
    selected_files = random.sample(all_files, num_files_to_select)

    # 将选定的文件复制到输出文件夹
    for file_name in all_files:
        if file_name in selected_files:
            file_path = os.path.join(input_folder, file_name)
            shutil.copy(file_path, output_folder_train)
        else:
            file_path = os.path.join(input_folder, file_name)
            shutil.copy(file_path, output_folder_val) 

    

# 用法示例，设置随机种子为100

def baseline2(readpath,savepath):
    list1=os.listdir(readpath)
    list1.sort(key=lambda x:int(x[:-4]))

    j=list1[0]
    i=0
    for d in list1:
        if j==d:
            df0=pd.read_csv(readpath+'\\'+j)
        else:
            df1=pd.read_csv(readpath+'\\'+d)
            if df0['SOC'].values[-1]-df1['SOC'].values[0]<0.2 and df1['elevation'].values[0]==df0['elevation'].values[-1] and abs(df1['Odometer'].values[0]-df0['Odometer'].values[-1])<0.1:
                df0=pd.concat([df0,df1])
                
            else:
                df0=df0.reset_index(drop=True)
                main_h(df0,d,str('train'))
                j=d
                df0=pd.read_csv(readpath+'\\'+j)
        i+=1
        df0.to_csv(savepath+'\\'+str(i)+'.csv',index=False)


# # 增强结果验证——xgboost，LightGBM，神经网络

# ## 五折交叉验证

# In[15]:


def validation(train_path,val_path,modelname,Validation_mode='Five fold'):

    list0=os.listdir(train_path)
    list1=os.listdir(val_path)
    if Validation_mode=='Five fold':
        df_result=pd.DataFrame()
        for i in range(5):
            train=pd.read_csv(train_path+'\\'+list0[i])
            train=traffic_feature1(train)
            val=pd.read_csv(val_path+'\\'+list1[i])
            val=traffic_feature1(val)
            if modelname=='LGB_1':
                feature=['mileage','drive_time','spdmean','bat_min','bat_max','weekday','lat','long','cur','volt','ECR']
                
                d_train=train[feature]
                
                d_val=val[feature]

                Lgb_para1=LGB_paras(d_train,10)

                train,val=LGB_model(d_train,d_val,Lgb_para1,10)

            elif modelname=='LGB_2':
                feature=['mileage','spdmean','Amb_temp','grad0','grad1','grad2','grad3','grad4','grad5','grad6','grad7','grad8','grad9','grad10','grad11','grad12','ECR']
              
                data_train=train[feature]

                data_val=val[feature]

                Lgb_para=LGB_paras(data_train,16)

                train,val=LGB_model(data_train,data_val,Lgb_para,16)

            elif modelname=='XGBoost':
                feature=['drive_time','mileage','tech_speed','acc_95','dec_5','Amb_temp','Weekdays morning rush hours','Weekdays evening rush hours','Weekdays non-rush hours','Weekends morning rush hours','Weekends evening rush hours','Weekends non-rush hours','ECR']

                df_train=train[feature]
                
                df_val=val[feature]

                XGB_para=XGB_paras(df_train,12)

                train,val=XGB_model(df_train, df_val,XGB_para,12)
                
            else:
                print('Undefined model')
                break

            df=pd.DataFrame()
            df['ture']=val['ECR']
            df[modelname]=val
            df_result=pd.concat([df_result,df],ignore_index=True)
            
    else:
        print('Wrong validation')
    return df_result

# Tesla model 3 rolling resistance parameters

para_a=179.40
para_b=0.2800
para_c=0.02350

# Other parameter settings for feature extraction


namelist=['grad'+str(i) for i in range(13)] 
beta_spd = 0.0007
# beta_spd = 1000000000
beta_m = 0.0001
para1=1919
para2 = [0.9*beta_spd,9.8,0.85,0.05,0.9*beta_m]


# Proposed data augmentation method applications
    
travelpath=r''  # Enter the address of travel data
                # such as travelpath=r'D:\Tesla model 3\train\original' 

trainsavepath=r''
valsavepath=r''

proposed(travelpath,trainsavepath,valsavepath)

# Baseline1 data augmentation method applications


save_baseline_1=r''
readtrain=r''
readval=r''
baseline_1(readtrain,readval,save_baseline_1)


# Baseline2 data augmentation method applications


input_folder = r''
output_folder_train = r''
output_folder_val = r''
random_select_files(input_folder, output_folder_train, output_folder_val, percentage=20)
baseline2(output_folder_train)



# Five fold cross validation
train_path=r''
val_path=r''


lgb1_val=validation(train_path,val_path,'LGB_1')
lgb2_val=validation(train_path,val_path,'LGB_2')
lgb2_val=validation(train_path,val_path,'XGBoost')

