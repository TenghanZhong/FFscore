import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  #t检验 from scipy import stats
import statsmodels.api as sm
import warnings
import os
import glob
import tushare as ts
from tqdm import tqdm
import re
import sys
from multiprocessing import Pool
warnings.filterwarnings("ignore")

def get_suspended_stocks(date):
    """
    获取指定日期的非停牌股票列表。
    :param date: 需要查询的日期，格式为 'YYYYMMDD'。
    :return: 非停牌股票的列表。list
    """
    # 尝试获取所有停牌股票信息
    try:
        suspend_info = pro.suspend(suspend_date=date, fields='ts_code,suspend_date')
        suspended_stocks = set(suspend_info['ts_code'])
    except Exception as e:
        print(f"Error fetching suspended stocks: {str(e)}")
        suspended_stocks = set()

    # 筛选出停牌股票
    suspended_stocks = list(suspended_stocks)

    return suspended_stocks


# 设置开始和结束日期
def get_last_monthly_trading_day(start_date,end_date):
    '''
    获取回测期间每个月最后一个交易日的列表
    :param start_date: 起始日  eg: start_date = '20060401'
    :param end_date: 终止日 eg: end_date = '20160401'
    :return: 返回list格式的回测期的每个last_trading_day
    '''

    # 获取交易日历数据
    trade_cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    # 筛选出开市的日子
    open_days = trade_cal[trade_cal['is_open']==1]
    # 将日期字符串转换为 datetime 对象，以便进行处理和筛选
    open_days['cal_date'] = pd.to_datetime(open_days['cal_date'])
    # 提取年份和月份，用于分组
    open_days['year'] = open_days['cal_date'].dt.year
    open_days['month'] = open_days['cal_date'].dt.month
    # 按年份和月份分组，并获取每个分组的最后一个个交易日
    last_trading_days = open_days.groupby(['year', 'month']).first().reset_index()
    last_trading_days['cal_date'] = last_trading_days['cal_date'].dt.strftime('%Y-%m-%d')#只提取年月日
    rebalancing_all_days=last_trading_days['cal_date'].tolist()
    return rebalancing_all_days  #返回list格式的last_trading_day

def get_first_monthly_trading_day(start_date,end_date):
    '''
    获取回测期间每个月最后一个交易日的列表
    :param start_date: 起始日  eg: start_date = '20060401'
    :param end_date: 终止日 eg: end_date = '20160401'
    :return: 返回list格式的回测期的每个last_trading_day
    '''

    # 获取交易日历数据
    trade_cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    # 筛选出开市的日子
    open_days = trade_cal[trade_cal['is_open']==1]
    # 将日期字符串转换为 datetime 对象，以便进行处理和筛选
    open_days['cal_date'] = pd.to_datetime(open_days['cal_date'])
    # 提取年份和月份，用于分组
    open_days['year'] = open_days['cal_date'].dt.year
    open_days['month'] = open_days['cal_date'].dt.month
    # 按年份和月份分组，并获取每个分组的最后一个个交易日
    last_trading_days = open_days.groupby(['year', 'month']).last().reset_index()
    last_trading_days['cal_date'] = last_trading_days['cal_date'].dt.strftime('%Y-%m-%d')#只提取年月日
    rebalancing_all_days=last_trading_days['cal_date'].tolist()
    return rebalancing_all_days  #返回list格式的last_trading_day

def get_first_yearly_trading_day(start_date,end_date):
    '''
    获取回测期间每年第一个交易日的列表
    :param start_date: 起始日  eg: start_date = '20060401'
    :param end_date: 终止日 eg: end_date = '20160401'
    :return: 返回list格式的回测期的每个first_trading_days
    '''

    # 获取交易日历数据
    trade_cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    open_days = trade_cal[trade_cal['is_open'] == 1]
    open_days['cal_date'] = pd.to_datetime(open_days['cal_date'])
    first_trading_days = open_days.groupby(open_days['cal_date'].dt.year).last().reset_index(drop=True)
    first_trading_days['cal_date'] = first_trading_days['cal_date'].dt.strftime('%Y-%m-%d')
    rebalancing_all_days = first_trading_days['cal_date'].tolist()
    return rebalancing_all_days


def get_fitst_trading_day_after_report_date(start_date, end_date):
    '''
    获取回测期间每个报告期的最后一个交易日的列表
    :param start_date: 起始日  eg: start_date = '20060401'
    :param end_date: 终止日 eg: end_date = '20160401'
    :return: 返回list格式的回测期的每个fitst_trading_day_after_report_date
    '''

    trade_cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    open_days = trade_cal[trade_cal['is_open'] == 1]
    report_dates = ['03-31', '06-30', '09-30', '12-31']
    rebalancing_days = []

    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for report_date in report_dates:
            date_str = pd.to_datetime(f"{year}-{report_date}")
            if date_str >= pd.to_datetime(start_date) and date_str <= pd.to_datetime(end_date):
                first_trade_date_after_report = open_days.loc[
                    pd.to_datetime(open_days['cal_date']) >= date_str, 'cal_date'].min()
                if not pd.isnull(first_trade_date_after_report):
                    formatted_date = pd.to_datetime(first_trade_date_after_report).strftime('%Y-%m-%d')
                    rebalancing_days.append(formatted_date)

    return rebalancing_days




def select_min_p_percent_stocks(group,column,percentile_or_value):
    """
    根据特定column的值从DataFrame中选择顶部p百分位的股票。

    :param group: 包含股票数据的DataFrame。
    :param column: 用于排序的列的名称。
    :param percentile: 要选择的股票的顶部百分比。
    :return: 包含顶部p百分位股票的DataFrame。
    """
    # 参数验证
    if not isinstance(group, pd.DataFrame):
        raise ValueError("group must be a pandas DataFrame")
    if not isinstance(column, str):
        raise ValueError("column must be a string")
    if not (0 < percentile_or_value <= 100):
        raise ValueError("percentile_or_value must be between 0 and 100")
    # 确认DataFrame包含所需列
    if column not in group.columns:
        raise ValueError(f"column {column} does not exist in the DataFrame")
    n = max(int(len(group) * (percentile_or_value / 100)), 1)#选取至少1支股票，
    return group.nsmallest(n, column)  # n个股票，按照Pb排序选择最小值

def select_top_fscore_stocks(group, column, percentile_or_value):
    '''

    :param group: dataframe形式，在get_rebalancing_day_stocks_datas后使用
    :param column:指标
    :param percentile_or_value:比例或值
    :return:dataframe
    '''
    mask=(group[column]==percentile_or_value)
    group=group[mask]
    return group

def process_single_file(file_path, date, pattern, suspended_stocks):
    '''

    :param file_path: 股票文件名
    :param date: 日期
    :param pattern:路径
    :param suspended_stocks:停牌股票的list
    :return: 非停牌股票且存在date当天数据的股票的那行数据 dataframe
    '''
    # 搜索文件路径，查找股票代码
    match = pattern.search(file_path)
    # 如果找到了匹配，提取股票代码
    if match:
        stock_code = match.group()
    else:
        print("No stock code found in the file path.")
        return None

    # 若为停牌，返回None
    if stock_code in suspended_stocks:
        return None

    try:
        data = pd.read_csv(file_path, compression='gzip')  # 使用 pandas 读取 gzip 压缩的 csv 文件
        if date in data['trade_date'].values:
            return data[data['trade_date'] == date]
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

    return None



def get_rebalancing_day_stocks_datas(files, date):
    '''

    :param files:文件夹路径
    :param date: 日期
    :return:返回当日存在数据且非停牌的所有股票数据的list
    '''

    # 创建一个正则表达式对象，匹配股票代码
    pattern = re.compile(r'\d{6}\.[A-Z]{2}')
    suspended_stocks = get_suspended_stocks(date)  # 当日停牌股票

    with Pool(processes=8) as pool:  # processes=核心数，[() for i in []]最后返回的是一个list[(),(),(),()......]
        #对list[(),(),(),()......]里每个()执行process_single_file函数,而processes=n是同时对n个()执行process函数
        results = pool.starmap(process_single_file, [(file_path, date, pattern, suspended_stocks) for file_path in files])

    # 筛选出有值的
    rebalancing_day_stocks_datas = [res for res in results if res is not None]

    return rebalancing_day_stocks_datas


def portfolio_one_period_pct(date_index,date,rebalance_stock_list,rebalancing_all_days,end_date):
    '''
    得到当期调仓日到下一调仓日每日投资组合的pct数据，返回一个df类型

    :param date_index:date的当日索引,循环时要用enumerate  date:遍历到的当天调仓日
    :param rebalance_stock_list:获取当期调仓日的调仓list，包含当期调仓的股票代码
    :param rebalancing_all_days:所有调仓日的列表
    :param end_date:回测期结束日
    :return:返回一个dataframe对象，得到当期调仓日到下一调仓日每日的pct数据
    '''

    stock_sta=[] #用来记录当前日期到下一调仓期所有持仓股票的每日数据  （最后用groupby得到每日投资组合pct）
    for ts_code in rebalance_stock_list:
        file_path1 = os.path.join(directory, f"{ts_code}.csv.gz")  # os.path.join()
        sta=pd.read_csv(file_path1, compression='gzip')
        if date_index + 1 < len(rebalancing_all_days):#注意索引从0开始
            # 如果不是最后一个元素，那么下一个调仓日就是列表中的下一个元素
            next_date = rebalancing_all_days[date_index + 1]
        else:
            # 如果是最后一个元素，那么可能没有“下一个”调仓日
            next_date = end_date
        mask = (sta['trade_date'] > date) & (sta[
            'trade_date'] <= next_date)  # 注意，date不能取当天，因为调仓日当天以close买入，下一天才有收益
        sta=sta[mask] #得到当前ts_code在这一period的所有数据
        stock_sta.append(sta)
    stock_sta=pd.concat(stock_sta, ignore_index=True)#股票池所有股票在period内所有数据
    stock_sta['trade_date'] = pd.to_datetime(stock_sta['trade_date'])#时间变换格式
    period_portfolio=pd.DataFrame(stock_sta.groupby('trade_date')['pct'].mean())#得到这一period 每一交易日投资组合的加权收益率（等额加权）
    period_portfolio['position']=np.ones(period_portfolio.shape[0])#表示持仓

    return period_portfolio


def portfolio_allstocks_one_period_return(date_index, date, rebalance_stock_list, rebalancing_all_days, end_date):
    '''
    得到当期调仓日到下一调仓日投资组合所包含所有股票的区间收益率数据，返回一个df类型\
         return	fscore	pb
    ts_code	990	  9	     5

    :param date_index:date的当日索引,循环时要用enumerate  date:遍历到的当天调仓日
    :param rebalance_stock_list:获取当期调仓日的调仓list，包含当期调仓的股票代码
    :param rebalancing_all_days:所有调仓日的列表
    :param end_date:回测期结束日
    :return:返回一个dataframe对象，得到当期调仓日到下一调仓日整个区间调仓股票的return，fscore，pb数据
    '''

    stock_sta = []  # 用来记录当前日期到下一调仓期所有持仓股票的每日数据  （最后用groupby得到每日投资组合pct）
    for ts_code in rebalance_stock_list:
        file_path1 = os.path.join(directory, f"{ts_code}.csv.gz")  # os.path.join()
        sta = pd.read_csv(file_path1, compression='gzip')
        if date_index + 1 < len(rebalancing_all_days):  # 注意索引从0开始
            # 如果不是最后一个元素，那么下一个调仓日就是列表中的下一个元素
            next_date = rebalancing_all_days[date_index + 1]
        else:
            # 如果是最后一个元素，那么可能没有“下一个”调仓日
            next_date = end_date
        mask = (sta['trade_date'] > date) & (sta[
                                                 'trade_date'] <= next_date)  # 注意，date不能取当天，因为调仓日当天以close买入，下一天才有收益
        sta = sta[mask]  # 得到当前ts_code在这一period的所有数据
        sta['net'] = (1 + sta['pct']).cumprod()
        if sta.empty:
            continue
        period_return = sta.iloc[-1]['net'] - 1
        stock_sta.append(
            pd.DataFrame({'return': [period_return], 'f_score': [sta.iloc[0]['f_score']], 'pb': [sta['pb'].mean()]
                             , 'log_mve': [np.log(sta['total_mv'].mean() * 10000)],
                          'log_pb': [np.log(sta['pb'].mean())]},
                         index=[ts_code]))
    stock_sta = pd.concat(stock_sta)  # 股票池所有股票在period内所有数据

    return stock_sta


def empty_pool(date_index,date,end_date,rebalancing_all_days):
    '''

    :param date_index:日期索引
    :param date:当前日期
    :param end_date:结束日期
    :param rebalancing_all_days:所有调仓日list
    :return:在判断当期空仓后，将投资组合的pct全部设为0，返回一个dataframe类型
    '''
    # 当没有符合选股条件时，空仓，pct=0
    if date_index + 1 < len(rebalancing_all_days):  # 注意索引从0开始
        # 如果不是最后一个元素，那么下一个调仓日就是列表中的下一个元素
        next_date = rebalancing_all_days[date_index + 1]
    else:
        # 如果是最后一个元素，那么可能没有“下一个”调仓日
        next_date = end_date
    trade_cal = pro.trade_cal(exchange='', start_date=date, end_date=next_date)
    open_days = trade_cal[trade_cal['is_open'] == 1]
    open_days['cal_date'] = pd.to_datetime(open_days['cal_date'])
    empty_portfolio_for_date = pd.DataFrame({'pct': np.zeros(len(open_days))}, index=open_days['cal_date'])
    empty_portfolio_for_date['position'] = np.zeros(empty_portfolio_for_date.shape[0])#表示空仓
    return empty_portfolio_for_date

def one_period_statistic(df,date):
    '''

    :param df: 需有'pct'的dataframe
    :return: 返回一个dataframe，记录当period的统计数据
    '''
    df['net'] = (1 + df['pct']).cumprod()
    df.reset_index(inplace=True)
    period_return= df.loc[df.shape[0] - 1, 'net']-1
    period_mean = df['pct'].mean()  # 计算相对收益的平均值
    period_median = df['pct'].median()
    quantile10 = df['pct'].quantile(0.1)
    quantile25 = df['pct'].quantile(0.25) #np.percentile()=pd.quantile()
    quantile75 = df['pct'].quantile(0.75)
    quantile90 = df['pct'].quantile(0.9)
    period_statistic = pd.DataFrame({
        'date' :[date],
        'return': [period_return],
        'Mean': [period_mean],
        'Median': [period_median],
        '10th Quantile': [quantile10],
        '25th Quantile': [quantile25],
        '75th Quantile': [quantile75],
        '90th Quantile': [quantile90]})
    return period_statistic

def all_periods_statistic(Portfolio_Daily_Summary,fscore):
    '''
    统计单个fscore整个回测区间的统计数据
    :param Portfolio_Daily_Summary: fscore在这个区间内的每月数据
    :param fscore:fscore值
    :return:dataframe，记录统计数据
    '''
    score_mean = Portfolio_Daily_Summary['return'].mean()  # 计算相对收益的平均值
    score_median = Portfolio_Daily_Summary['return'].median()
    quantile10 = Portfolio_Daily_Summary['return'].quantile(0.1)
    quantile25 = Portfolio_Daily_Summary['return'].quantile(0.25)
    quantile75 = Portfolio_Daily_Summary['return'].quantile(0.75)
    quantile90 = Portfolio_Daily_Summary['return'].quantile(0.90)
    positive_percent = (Portfolio_Daily_Summary['return'] > 0).sum() / (Portfolio_Daily_Summary['return'].shape[0])
    fscore_statistic=pd.DataFrame({
        'Mean': [score_mean],
        'Median': [score_median],
        '10th Quantile': [quantile10],
        '25th Quantile': [quantile25],
        '75th Quantile': [quantile75],
        '90th Quantile': [quantile90],
        '正值占比': ['{}%'.format(positive_percent * 100)]
    }, index=['{}'.format(fscore)]) # pd.dataframe({'a':[],'b':[]})
    return fscore_statistic



def bootstrap_test(all_returns,winner_returns, loser_returns, n_bootstraps=1000):
    """
    使用Bootstrap方法来估计Winner组合与Loser组合之间收益差异的显著性。
    在本实验中:bootstrap方法是对所有样本随机抽取两个组合，计算两个组合各个统计指标的差值
    用这个差值序列和实际值actual值做t检验，p＜0.05则说明，实际差值与随机组合的差值有差异
    即winner-loser的差值是显著存在的，不是随机游动的，则我们可以认为winner组合和loser组合收益有明显差异

    参数:
    all_returns:包含所有组合的数据dataframe，不只是winner和loser
    winner_returns: dataframe，包含Winner组合的所有数据。
    loser_returns: dataframe，包含Loser组合的所有数据。
    n_bootstraps: int，Bootstrap样本的数量，默认为1000。

    返回:
    一个字典，包含不同统计量（均值、中位数等）的P-value。
    # 示例使用：
    # p_values = bootstrap_test(winner_returns, loser_returns)
    # for percentile, p_value in p_values.items():
    # print(f"{percentile.capitalize()} P-value: {p_value:.4f}")
    """
    # 确定Winner和Loser组合的样本大小
    n_winner = winner_returns.shape[0]
    n_loser = loser_returns.shape[0]

    # 存储Bootstrap过程中计算的差值
    bootstrap_diffs = {
        'means': [],
        'medians': [],
        '10th': [],
        '25th': [],
        '75th': [],
        '90th': []
    }

    # 实际观察到的差异
    actual_diffs = {
        'means': [winner_returns['return'].mean() - loser_returns['return'].mean()],
        'medians': [winner_returns['return'].median() - loser_returns['return'].median()],
        '10th':[winner_returns['return'].quantile(0.10) - loser_returns['return'].quantile(0.10)],
        '25th': [winner_returns['return'].quantile(0.25) - loser_returns['return'].quantile(0.25)],
        '75th': [winner_returns['return'].quantile(0.75) - loser_returns['return'].quantile(0.75)],
        '90th': [winner_returns['return'].quantile(0.9) - loser_returns['return'].quantile(0.90)]
    }

    # Bootstrap过程
    for _ in range(n_bootstraps):
        # 从所有样本中随机挑选样本
        boot_sample = all_returns.loc[np.random.choice(all_returns.index, size=(n_winner + n_loser), replace=True)]
        # 将随机挑选的样本按照PB分配到Winner和Loser组合
        boot_sample=boot_sample.sort_values(by='pb')#从小到大排
        boot_winner = boot_sample[:n_winner]['return'].values#换成np数组形式，便于后面求np.mean,np.percentile等
        boot_loser = boot_sample[n_winner:]['return'].values
        # 计算并存储不同百分位数的差值
        for stat in bootstrap_diffs.keys():
            if stat == 'means':
                bootstrap_diffs['means'].append(np.mean(boot_winner) - np.mean(boot_loser))
            elif stat == 'medians':
                bootstrap_diffs['medians'].append(np.median(boot_winner) - np.median(boot_loser))
            else:
                percentile = int(stat.rstrip('th'))
                bootstrap_diffs[stat].append(np.percentile(boot_winner, percentile) - np.percentile(boot_loser, percentile))

    # 计算P-value
    p_values = {}
    for stat, diffs in bootstrap_diffs.items():#dict_items([('means', [0.17394611526489068，0.21321421,0.8433123123,.......]), ('medians', [0.0738684563187394]),.......]
        observed_stat = actual_diffs[stat][0]
        diffs = np.array(diffs)
        # 计算t统计量
        t_stat = (observed_stat - np.mean(diffs)) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        # 计算两侧p-value
        p_values[stat] = stats.t.sf(np.abs(t_stat), len(diffs) - 1) * 2  # 两侧检验  相当于查表：stats.t.sf（t统计量,自由度）

    return p_values


##回测实现：↓
def get_portfolio_backtest_periods_pct(start_date, end_date, directory,rebalancing_method,strategy, column,percentile_or_value):
    '''
    总函数，获得整个回测期资产组合的每日pct的dataframe
    :param start_date:回测开始日
    :param end_date:回测结束日
    :param directory:文件路径
    :param rebalancing_method:调仓方法（月调还是年调etc)
    :param strategy:选股方法
    :param column:选股指标
    :param percentile:百分比
    :return:得到的就是整个回测期这个fscore每个区间的统计数据表dataframe
    '''
    # 获取调仓日期
    rebalancing_all_days = rebalancing_method(start_date, end_date)  # 回测期内每个月调仓日期
    # 设置文件路径
    # 查找所有的 .csv.gz 文件
    csv_files = glob.glob(os.path.join(directory, '*.csv.gz'))
    Portfolio_Daily_Summary_list = []  # 总表，用来记录每日投资组合的pct
    Portfolio_list = []  # 每次调仓list的加总
    statistic=[]#统计数据
    one_month_sample=[]#用作bootstrap检验，记录当前fscore下所有1月股票样本（记录了当前fscore下股票池每个股票1月收益，fscore和pb）

    # 遍历文件并读取内容
    for date_index, date in tqdm(enumerate(rebalancing_all_days), desc='backtesting', total=len(rebalancing_all_days)):
        rebalancing_day_stocks_datas = get_rebalancing_day_stocks_datas(csv_files, date)#这个步骤中筛除了当日停牌股
        if len(rebalancing_day_stocks_datas) == 0:# 判断调仓日是否没有股票符合选股策略条件
            print('调仓日：{}无数据'.format(date))
            continue
        rebalancing_day_stocks_datas = pd.concat(rebalancing_day_stocks_datas, ignore_index=True)  # 只有stock_pool不为空才继续
        stock_pool = strategy(rebalancing_day_stocks_datas, column, percentile_or_value)# 筛选出pb最小的百分之percentile的股票
        if len(stock_pool) == 0:
            print('调仓日：{}没有股票符合选股条件'.format(date))
            empty_portfolio_for_date = empty_pool(date_index,date,end_date,rebalancing_all_days)
            Portfolio_Daily_Summary_list.append(empty_portfolio_for_date)
            continue
        rebalance_stock_list = stock_pool['ts_code'].tolist()  # 获取调仓list

        Portfolio_list.append(rebalance_stock_list)  # 调仓列表加总
        period_portfolio = portfolio_one_period_pct(date_index, date, rebalance_stock_list, rebalancing_all_days,
                                                           end_date)  # 投资组合在period内每天的pct收益
        print('日期：{}, 调仓前五只为：{}'.format(date, rebalance_stock_list[:5]))

        ###每个period投资组合的收益率 用于统计分析检验 1 月相对收益的平均值、中位数、10、 25、 75、90 分位数数值,正收益的样本数占比，以及样本数
        period_portfolio_df = period_portfolio
        statistic.append(one_period_statistic(period_portfolio_df,date))

        ###每个period中调仓的所有股票的这一period的总收益率，fscore和pb
        portfolio_allstocks_one_period = portfolio_allstocks_one_period_return(date_index, date,
                                                                                      rebalance_stock_list,
                                                                                      rebalancing_all_days,
                                                                                      end_date)
        one_month_sample.append(portfolio_allstocks_one_period)

    if len(statistic)==0 or len(one_month_sample)==0:
        return None,None

    else:
        all_periods_one_month_sample=pd.concat(one_month_sample,axis=0)
        statistic=pd.concat(statistic,axis=0)#将每个区间内的统计数据连接，得到的就是整个回测期这个fscore每个区间的统计数据表
        return statistic,all_periods_one_month_sample  # 返回整个回测周期内每个月的统计数据汇总

#基础数据↓

if __name__ == '__main__':
    # 用你的token初始化tushare，你可以在注册后通过Tushare网站获取token
    ts.set_token('48c54212788b6a040d89de4ee5810744d936b44c2423302761f3b254')  # 请替换成你自己的token
    pro = ts.pro_api()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
    plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of minus sign '-' display as square
    # 基础数据
    start_date = '20130401'  #回测期开始日
    end_date = '20160401'   #回测期结束日
    directory ='D:\\Stock_data_fscore'  ##

    fscore_statistic=[]#记录所有fscore的平均数据
    record_highscore=[]#记录单个fscore的所有月度数据,fscore=8或9
    record_lowscore = []  # 记录单个fscore的所有月度数据,fscore=0或1
    record_all=[] # 记录所有fscore的所有月度数据
    all_fscores_one_month_sample=[]#记录所有股票1月样本
    # 投资组合各个fscore的dataframe
    for fscore in tqdm(range(0,10), desc='fscore', total=len(range(0,10))):
        print('fscore={}'.format(fscore))
        Portfolio_Daily_Summary,all_periods_one_month_sample = get_portfolio_backtest_periods_pct(start_date=start_date, end_date=end_date, directory=directory,
                                                                     rebalancing_method=get_first_monthly_trading_day,
                                                                     strategy=select_top_fscore_stocks,column='f_score',percentile_or_value=fscore)
        if Portfolio_Daily_Summary is not None and not Portfolio_Daily_Summary.empty:
            all_fscores_one_month_sample.append(all_periods_one_month_sample)
            fscore_statistic.append(all_periods_statistic(Portfolio_Daily_Summary, fscore))
            record_all.append(Portfolio_Daily_Summary)
            if fscore==0 or fscore==1:
                record_lowscore.append(Portfolio_Daily_Summary)
            elif fscore==8 or fscore==9:
                record_highscore.append(Portfolio_Daily_Summary)

    all_fscores_one_month_sample=pd.concat(all_fscores_one_month_sample)
    fscore_statistic= pd.concat(fscore_statistic)
    record_highscore=pd.concat(record_highscore)
    record_lowscore = pd.concat( record_lowscore)
    record_all = pd.concat( record_all)

    record_all_ptest=pd.DataFrame(record_all.groupby('date')['return'].mean())#做groupby，把其中每个fscore在相同日期的return数据取均值合并
    record_highscore_ptest=pd.DataFrame(record_highscore.groupby('date')['return'].mean())

    record_lowscore_sta = all_periods_statistic(record_lowscore, fscore='LowScore')#计算整个回测期的统计数据，变成一行dataframe
    record_highscore_sta =all_periods_statistic(record_highscore, fscore='HighScore')
    record_all_sta = all_periods_statistic(record_all, fscore='AllScore')

    # 进行t检验 和 P检验（wilcoxon检） 和二项检验(binom_test)

    # Two-sample t-test
    t_stat_high_all, p_val_high_all = stats.ttest_ind(record_highscore['return'], record_all['return'])


    # Signed rank Wilcoxon test
    record_high_all_ptest = record_highscore_ptest.join( record_all_ptest,lsuffix='_highscore',rsuffix='_allscore',how='inner')
    w_stat, p_value = stats.wilcoxon( record_high_all_ptest['return_highscore'],  record_high_all_ptest['return_allscore'])
    # 注意，使用wilcoxon检验要求样本长度一致 在这里这个检验的意义在于区分高fscore组和全fscore组的return中位数在每个时间段是否不同
    # 此检验假设数据是成对的，即每个High FScore的数据点都与All FScore的一个数据点相对应
    # 这种配对基于相同的时间点或相同的股票（这里是时间点，因为不同fscore的股票池不一样）
    # 由于Wilcoxon检验对数据分布的要求较低，即使收益数据不符合正态分布，使用此检验也是合理的
    # 此外，考虑到股票收益的潜在非对称性和异常值，Wilcoxon检验可能比传统的t检验更有统计效能

    #二项检验
    positive_high = record_highscore['return'][ record_highscore['return'] > 0].count()#是“High FScore”组合中正收益股票的数量。
    positive_all =  record_all['return'][ record_all['return'] > 0].count()# positive_all是“High FScore”组合中正收益股票的数量。
    binom_p_value = stats.binom_test(positive_high,n=record_highscore.shape[0],
                                     p=positive_all / record_all.shape[0]) #p为“All FScore”组合中正收益股票的比例，这里作为成功的概率使用。
    # 使用二项检验来确定High FScore组合中正收益股票的比例是否显著高于在All FScore组合中观察到的正收益股票的比例。
    # 这里，n 是High FScore组合中股票的总数，p 是All FScore组合中正收益股票的比例，用作High FScore组合中正收益股票比例的期望概率。
    # 如果二项检验的P值小于显著性水平（如0.05），则我们可以认为High FScore组合中正收益的比例与All FScore组合存在显著差异，这可能表明高FScore评分的股票实际上有更好的表现。

    #bootstrap检验  |表示or
    mask_win = (all_fscores_one_month_sample['f_score'] == 8) | (all_fscores_one_month_sample['f_score'] == 9)
    mask_lose = (all_fscores_one_month_sample['f_score'] == 0) | (all_fscores_one_month_sample['f_score'] == 1)
    winner_returns=all_fscores_one_month_sample[mask_win]#转变为numpy数组形式，.to_numpy()  .values返回的是np数组
    loser_returns=all_fscores_one_month_sample[ mask_lose]
    bootstrap_p_value=bootstrap_test(all_returns=all_fscores_one_month_sample,winner_returns=winner_returns,loser_returns= loser_returns)

    #计算指标的相关系数
    spearman_corr=pd.DataFrame({'one_month_return':[all_fscores_one_month_sample['return'].corr(
        all_fscores_one_month_sample['f_score'],method='spearman')]},index=['f_score'])
    pearson_corr = pd.DataFrame({ 'one_month_return': [ all_fscores_one_month_sample['return'].corr(
        all_fscores_one_month_sample['f_score'], method='pearson')]}, index=['f_score'])
    #当你关心的是变量之间的严格线性关系，并且数据满足正态分布时，应该使用皮尔森相关系数。
    # 如果你的数据不满足正态分布，或者你只关心变量之间的单调关系（不一定是线性的），则应该使用斯皮尔曼相关系数。
    # 在处理金融数据时，由于它们常常包含离群值并且分布可能远离正态，斯皮尔曼相关系数可能更加适用。

    ##回归得到估计因子暴露
    x = sm.add_constant(all_fscores_one_month_sample[['log_mve', 'log_pb', 'f_score']])
    y = df['return']

    model = sm.OLS(y, x).fit()
    model_summary = model.summary()

    # 设置Pandas的显示选项以显示更多列
    pd.set_option('display.max_columns', None)
    # 设置显示宽度以适应上述列的显示，这里可以设置一个足够大的数字
    pd.set_option('display.width', 1000)
    fscore_statistic = pd.concat([fscore_statistic, record_highscore_sta, record_lowscore_sta, record_all_sta], axis=0)
    print(fscore_statistic)
    print('对highscore和allscore进行统计检验：t检验值为{},Wilcoxon P检验值为{},二项检验值为{}'.format(
        t_stat_high_all,p_value, binom_p_value))
    print('bootstrap检验的p值为{}'.format( bootstrap_p_value))
    print('未来一月收益和fscore的spearman系数为{},未来一月收益和fscore的pearson系数为{}'.format( spearman_corr, pearson_corr))
    print(model_summary)

'''
fscore方案改进意见，pca主成分法
'''
