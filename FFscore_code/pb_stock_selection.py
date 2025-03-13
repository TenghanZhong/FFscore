import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  #t检验 from scipy import stats
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Microsoft YaHei'  # 或者其他支持中文的字体名称
plt.rcParams['axes.unicode_minus'] = False  # 这句话是为了让负号显示正常

data=pd.read_csv(r'D:\Stock_data_zscore\pb_zscore_relationships.csv')
code_list=pd.read_csv(r'D:\Stock_data_zscore\code_list.csv')

data['flag'] = 0  # 买卖标记，买入：1，卖出：-1
data['position'] = 0  # 持仓状态，持仓：1，不持仓：0
data['trade_times'] = 0  # 统计交易次数

#按市净率分层  注意：这里交易策略是在每个报告期，按照pb筛选出5个区间，分别作为5个投资组合进行购买，持有到下一个报告期，根据下一个报告期排序来调仓
# 因此每个报告期都要筛选一次--日期，pb,code
data['end_date']=pd.to_datetime(data['end_date'])
datetime_objects= [
    "2006/9/30", "2006/12/31", "2007/3/31", "2007/6/30", "2007/9/30",
    "2007/12/31", "2008/3/31", "2008/6/30", "2008/9/30", "2008/12/31",
    "2009/3/31", "2009/6/30", "2009/9/30", "2009/12/31", "2010/3/31",
    "2010/6/30", "2010/9/30", "2010/12/31", "2011/3/31", "2011/6/30",
    "2011/9/30", "2011/12/31", "2012/3/31", "2012/6/30", "2012/9/30",
    "2012/12/31", "2013/3/31", "2013/6/30", "2013/9/30", "2013/12/31",
    "2014/3/31", "2014/6/30", "2014/9/30", "2014/12/31", "2015/3/31",
    "2015/6/30", "2015/9/30", "2015/12/31", "2016/3/31"
]
# 使用pandas的to_datetime方法转换日期字符串
period = pd.to_datetime(datetime_objects)

data['next_close'] = data.groupby('ts_code')['close'].shift(-1)#groupby相同相同code的代码，将下一期的close应用在上一期的next_close上

# 构造函数，选取前百分之20%pb的股票
def select_top_20_percent_stocks_with_next_close(group):
    n = max(len(group) // 5, 1)#选取至少1支股票，
    return group.nsmallest(n, 'pb')  # 选取前百分之20%股票

# 在不同的end_date中应用”选取函数“,来构建每个日期的股票池，同时这会让end_date变为index
selected_stocks_20_percent = data.groupby('end_date').apply(select_top_20_percent_stocks_with_next_close)

# 计算当期收益率
selected_stocks_20_percent['return'] = (selected_stocks_20_percent['next_close'] - selected_stocks_20_percent['close']) / selected_stocks_20_percent['close']
selected_stocks_20_percent.rename(columns={'end_date': 'trade_date'}, inplace=True)

# 删除NAN，即最后一个报告期的行
selected_stocks_20_percent = selected_stocks_20_percent.dropna(subset=['return'])

# 每个报告期计算收益率均值
mean_returns_20_percent = selected_stocks_20_percent.groupby('trade_date')['return'].mean()
# 计算净值
cumulative_return_20_percent = (1 + mean_returns_20_percent).cumprod()

# 计算总收益率
final_cumulative_return_20_percent_corrected = (cumulative_return_20_percent.iloc[-1]-1) * 100  # Convert to percentage

#计算最大回撤
max_drawdown = ((cumulative_return_20_percent.cummax()-cumulative_return_20_percent)/cumulative_return_20_percent.cummax()).max()

annual_ret = 100 * (pow(1 + cumulative_return_20_percent.iloc[-2],
                        250 / (cumulative_return_20_percent.shape[0]*3*30) )- 1)

print(f'最大回撤率：{round(max_drawdown * 100, 2)}%')
print('收益率为{}%'.format(final_cumulative_return_20_percent_corrected))
print('策略的年化收益率：%.2f%%' % (annual_ret))


fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.plot(cumulative_return_20_percent)
ax1.set_xlabel('时间')
ax1.set_ylabel('净值')
ax1.set_title('pb策略净值曲线')
plt.show()
