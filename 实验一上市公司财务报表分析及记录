#coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm
import  matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# 加载数据
data = pd.read_csv('E1.csv')
data.rename(columns={'下一年净资产收益率': 'y',
                     '资产周转率': 'x1',
                     '利润率': 'x2',
                     '债务资本比率': 'x3',
                     '成长速度(%)': 'x4',
                     '市倍率': 'x5',
                     '收入质量': 'x6',
                     '存货率': 'x7',
                     '资产规模': 'x8',
                     '当年净资产收益率': 'x9'},
            inplace=True)

# 描述性统计
desc = data.describe()
print(desc)

X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']]
y = data['y']

# 向X添加常数项
X = sm.add_constant(X)

# 创建模型并拟合数据
model = sm.OLS(y, X).fit()

# 输出回归结果
print(model.summary())
#desc.to_csv('desc_tabel.csv')
