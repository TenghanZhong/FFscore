import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  #t检验 from scipy import stats
import warnings
warnings.filterwarnings("ignore")



plt.rcParams['font.family'] = 'Microsoft YaHei'  # 或者其他支持中文的字体名称
plt.rcParams['axes.unicode_minus'] = False  # 这句话是为了让负号显示正常

data_rela=pd.read_csv(r'D:\Stock_data_zscore\pb_zscore_relationships.csv')

#按市净率分层
data_rela=data_rela.sort_values(by='pb')
data_rela['quantile']=pd.qcut(data_rela['pb'],5,labels=False)#分为5层，01234  0位pb最小 4为最大
#这里先sort了，所以不用labels

data_pb0,data_pb1,data_pb2,data_pb3,data_pb4=data_rela.query('quantile==0'),data_rela.query('quantile==1'),data_rela.query('quantile==2'
                                                    ),data_rela.query('quantile==3'),data_rela.query('quantile==4')

#画各层Pb与Zscore关系图
fig, axs = plt.subplots(3, 2, figsize=(18, 8))  # 3 rows, 2 columns

# 数据集和标题列表
datasets = [data_pb0, data_pb1, data_pb2, data_pb3, data_pb4]
titles = ['pb(0-20%) vs. zscore', 'pb(20%-40%) vs. zscore', 'pb(40%-60%) vs. zscore', 'pb(60%-80%) vs. zscore', 'pb(80%-100%) vs. zscore']
col=['pb(0-20%)', 'pb(20%-40%) ', 'pb(40%-60%)', 'pb(60%-80%)', 'pb(80%-100%)']

# 遍历所有的子图和数据集
for i, ax in enumerate(axs.flat):  # 使用 axs.flat 来获得所有子图的迭代器
    if i < len(datasets):  # 只在有数据的情况下绘制子图
        ax.scatter(datasets[i]['pb'], datasets[i]['z_score'], c='b')
        ax.set_xlabel('pb')
        ax.set_ylabel('zscore')
        ax.set_title(titles[i])
    else:
        ax.axis('off')  # 如果没有数据，隐藏多余的子图

plt.tight_layout()  # 调整子图的位置，以防止它们重叠
plt.show()

# 市净率分层下的 Z-Score 分布统计值
data_list=[]
pd.options.display.float_format = '{:.2f}'.format#设置浮点数表达
for data,i in zip(datasets,col):  #双重遍历
    data_zmean=round(data['z_score'].mean(),2)
    data_zstd = round(data['z_score'].std(),2)
    data_zcorr = round(data['z_score'].corr(data['pb']),2)
    #按照Z_score分状态
    bins = [-101, 1.81, 2.675, 301]
    labels = ['Financial distress', 'Grey area', 'Strong financial position']
    data['position'] = pd.cut(data['z_score'], bins=bins, labels=labels)

    grey_area=data[data['position']=='Grey area'].shape[0]#灰色区域样本数
    Financial_distress=round(data[data['position']=='Financial distress'].shape[0],0)#财务危机样本数
    Financial_distress_quantile=round(Financial_distress/data.shape[0],2)#财务危机占比

    #t检验
    t_stat, p_value = stats.ttest_ind(data['z_score'], data['pb'])#stats.ttest_ind
    data_statistic=pd.DataFrame({'Z-Score 平均数':[data_zmean],
                            'Z-Score 标准差':[data_zstd],
                            'Z-Score 相关系数':[data_zcorr],
                            'T 检验':[t_stat],
                            'T 检验p值(<0.05)':[p_value],
                            '灰色区域样本数':[grey_area],
                            '财务困境样本数':[Financial_distress],
                            '财务困境样本占比':[Financial_distress_quantile]})
    data_statistic = data_statistic.transpose()#转置
    data_statistic.rename(columns={0: '{}'.format(i)}, inplace=True)#更改列名
    data_list.append(data_statistic)
data_sta=pd.concat(data_list, axis=1)#横向链接
print(data_sta)

#画分层集中散点图
# 设置seaborn的风格
sns.set(style="whitegrid")

# 使用stripplot绘制散点图 sns.stripplot(x=, y=, data= ,palette=自适应颜色)
plt.figure(figsize=(18, 8))
sns.stripplot(x='quantile', y='z_score', data=data_rela, jitter=True,palette='viridis', size=5)

# 设置标题和坐标轴标签
plt.title("Z_SCORE by PB Quantile")
plt.xlabel("PB Quantile")
plt.ylabel("Z_SCORE")
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['pb0-20%', 'pb20%-40%', 'pb40%-60%', 'pb60%-80%', 'pb80%-100%'])

plt.show()
