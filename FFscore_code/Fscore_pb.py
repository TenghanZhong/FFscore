import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  #t检验 from scipy import stats
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


def get_st_stocks(date):
    """
    获取指定日期的ST股票列表。
    :param date: 需要查询的日期，格式为 'YYYYMMDD'。
    :return: ST股票的列表。list
    """

    # 尝试获取所有股票的基本信息
    try:
        # 获取当前所有上市股票的列表
        stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        # 筛选出ST股票
        st_stocks = stock_info[stock_info['name'].str.contains('ST')]
        st_stocks_list = st_stocks['ts_code'].tolist()
    except Exception as e:
        print(f"Error fetching ST stocks: {str(e)}")
        st_stocks_list = []

    return st_stocks_list



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
    code_list=group.nsmallest(n, column)
    code_list=code_list['ts_code'].tolist()
    return  code_list # n个股票，按照Pb排序选择最小值

def select_low_total_mv_stocks(group,column,percentile_or_value,date):

    mask = (group[column] > 20000) & (group[column] < 100000) & (group['trade_date'] == date)
    list=group[mask]['ts_code'].tolist()
    buy_list=[]
    for code in list:
        stock_data=group[group['ts_code']==code]
        if (stock_data.loc[stock_data['trade_date']==date,column]).any() < (stock_data[column].mean()):
            buy_list.append(code)

    return buy_list


def select_top_fscore_stocks(group, column, percentile_or_value):
    '''

    :param group: dataframe形式，在get_rebalancing_day_stocks_datas后使用
    :param column:指标
    :param percentile_or_value:比例或值
    :return:dataframe
    '''
    mask=(group[column]>=percentile_or_value)
    group=group[mask]['ts_code'].tolist()
    return group

def process_single_file(file_path, date, pattern, suspended_stocks,st_stocks):
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
    if (stock_code in suspended_stocks) or (stock_code in st_stocks):
        return None

    try:
        data = pd.read_csv(file_path, compression='gzip')  # 使用 pandas 读取 gzip 压缩的 csv 文件
        if date in data['trade_date'].values:
            return data[data['trade_date'] == date]
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

    return None

def process_single_file_before_rebalancing_day(file_path, date, pattern, suspended_stocks,st_stocks):
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
    if (stock_code in suspended_stocks) or (stock_code in st_stocks):
        return None

    try:
        data = pd.read_csv(file_path, compression='gzip')  # 使用 pandas 读取 gzip 压缩的 csv 文件
        if date in data['trade_date'].values:
            return data[data['trade_date'] <= date]
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
    st_stocks=get_st_stocks(date)#当日St股票

    with Pool(processes=8) as pool:  # processes=核心数，[() for i in []]最后返回的是一个list[(),(),(),()......]
        #对list[(),(),(),()......]里每个()执行process_single_file函数,而processes=n是同时对n个()执行process函数
        results = pool.starmap(process_single_file, [(file_path, date, pattern, suspended_stocks,st_stocks) for file_path in files])

    # 筛选出有值的
    rebalancing_day_stocks_datas = [res for res in results if res is not None]

    return rebalancing_day_stocks_datas

def get_before_rebalancing_day_stocks_datas(files, date):
    '''

    :param files:文件夹路径
    :param date: 日期
    :return:返回当日存在数据且非停牌的所有股票数据的list
    '''

    # 创建一个正则表达式对象，匹配股票代码
    pattern = re.compile(r'\d{6}\.[A-Z]{2}')
    suspended_stocks = get_suspended_stocks(date)  # 当日停牌股票
    st_stocks = get_st_stocks(date)  # 当日St股票

    with Pool(processes=8) as pool:  # processes=核心数，[() for i in []]最后返回的是一个list[(),(),(),()......]
        # 对list[(),(),(),()......]里每个()执行process_single_file函数,而processes=n是同时对n个()执行process函数
        results = pool.starmap(process_single_file_before_rebalancing_day,
                               [(file_path, date, pattern, suspended_stocks, st_stocks) for file_path in files])

    # 筛选出有值的
    rebalancing_day_stocks_datas = [res for res in results if res is not None]

    return rebalancing_day_stocks_datas

def portfolio_one_period_pct(date_index,date,rebalance_stock_list,rebalancing_all_days,end_date):
    '''
    得到当期调仓日到下一调仓日每日投资组合的pct数据，返回一个dataframe类型

    :param date_index，date的当日索引,循环时要用enumerate  date:遍历到的当天调仓日
    :param rebalance_stock_list:获取当期调仓日的调仓list，包含当期调仓的股票代码
    :param rebalancing_all_days:所有调仓日的列表
    :param end_date:回测期结束日
    :return:返回一个dataframe对象，得到当期调仓日到下一调仓日每日的pct数据和position持仓情况
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
        if not sta.empty:
            period_return = sta.iloc[-1]['net'] - 1
        else:
            # 处理空的DataFrame的情况，例如可以将period_return设置为某个默认值
            period_return = 0
        stock_sta.append(
            pd.DataFrame({'return': [period_return]},
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
    date=pd.to_datetime(date).strftime('%Y%m%d')
    next_date = pd.to_datetime(next_date).strftime('%Y%m%d')
    trade_cal = pro.trade_cal(exchange='', start_date=date, end_date=next_date)
    open_days = trade_cal[trade_cal['is_open'] == 1].sort_values(by='cal_date')###一定要记住，trade_cal给的日期顺序是倒序，所以要sort_values
    open_days['cal_date'] = pd.to_datetime(open_days['cal_date'])
    open_days.rename(columns={'cal_date': 'trade_date'}, inplace=True)
    empty_portfolio_for_date = pd.DataFrame({'pct': np.zeros(len(open_days))}, index=open_days['trade_date'])
    empty_portfolio_for_date['position'] = np.zeros(empty_portfolio_for_date.shape[0])#表示空仓
    return empty_portfolio_for_date

###主函数：↓
def get_portfolio_backtest_periods_pct(start_date, end_date, directory,rebalancing_method,select_strategy, column,percentile_or_value):
    '''
    总函数，获得整个回测期资产组合的每日pct的dataframe
    :param start_date:回测开始日
    :param end_date:回测结束日
    :param directory:文件路径
    :param rebalancing_method:调仓方法（月调还是年调etc)
    :param select_strategy:选股方法
    :param column:选股指标
    :param percentile:百分比
    :return:资产组合的每日pct的dataframe
    '''
    # 获取调仓日期
    rebalancing_all_days = rebalancing_method(start_date, end_date)  # 回测期内每个月调仓日期
    # 设置文件路径
    # 查找所有的 .csv.gz 文件
    csv_files = glob.glob(os.path.join(directory, '*.csv.gz'))#找到directory下所有.csv.gz结尾的文件
    Portfolio_Daily_Summary_list = []  # 总表，用来记录每日投资组合的pct
    Portfolio_list = []  # 每次调仓list的加总
    one_month_sample = []#得到区间内投资组合所包含的所有股票的区间数据

    # 遍历文件并读取内容
    for date_index, date in tqdm(enumerate(rebalancing_all_days), desc='backtesting', total=len(rebalancing_all_days)):
        rebalancing_day_stocks_datas = get_before_rebalancing_day_stocks_datas(csv_files, date)#这个步骤中筛除了当日停牌股
        if len(rebalancing_day_stocks_datas) == 0:# 判断调仓日是否存在股票数据
            print('调仓日：{}无数据'.format(date))
            continue
        rebalancing_day_stocks_datas = pd.concat(rebalancing_day_stocks_datas, ignore_index=True)  # 只有stock_pool不为空才继续
        stock_pool = select_strategy(rebalancing_day_stocks_datas, column, percentile_or_value,date)# 筛选出pb最小的百分之percentile的股票
        if len(stock_pool) == 0:
            print('调仓日：{}没有股票符合选股条件'.format(date))
            empty_portfolio_for_date = empty_pool(date_index,date,end_date,rebalancing_all_days)
            Portfolio_Daily_Summary_list.append(empty_portfolio_for_date)
            continue
        rebalance_stock_list = stock_pool# 获取调仓list

        Portfolio_list.append(rebalance_stock_list)  # 调仓列表加总
        period_portfolio = portfolio_one_period_pct(date_index, date, rebalance_stock_list, rebalancing_all_days,
                                                           end_date)  # 投资组合在period内每天的pct收益
        print('日期：{}, 调仓前五只为：{}'.format(date, rebalance_stock_list[:5]))
        Portfolio_Daily_Summary_list.append(period_portfolio)#添加到总表当中,导入总表的数据index为日期

        ###每个period的收益率
        period_portfolio_df = period_portfolio.copy()#已经是dataframe格式
        period_portfolio_df.reset_index(inplace=True)#原来的索引是日期，在统计区间段收益时reset一下，总表之前已经导入
        period_portfolio_df['Net_Asset_Value'] = (1 + period_portfolio_df['pct']).cumprod()  # 计算当期净值
        print('当期内收益为{}%'.format(
            (period_portfolio_df.loc[period_portfolio_df.shape[0] - 1, 'Net_Asset_Value'] - 1) * 100))

        ###每个period中调仓的所有股票的这一period的总收益率，fscore和pb
        portfolio_allstocks_one_period = portfolio_allstocks_one_period_return(date_index, date,
                                                                               rebalance_stock_list,
                                                                               rebalancing_all_days,
                                                                               end_date)
        one_month_sample.append(portfolio_allstocks_one_period)

    one_month_sample = pd.concat(one_month_sample)
    Portfolio_Daily_Summary = pd.concat(Portfolio_Daily_Summary_list)#将每个调仓期间内的dataframe连接
    Portfolio_Daily_Summary.index = pd.to_datetime(Portfolio_Daily_Summary.index)#将时间换成to_datetime格式

    return Portfolio_Daily_Summary,one_month_sample  # 返回投资组合每日pct的dataframe

#基础数据↓

if __name__ == '__main__':
    # 用你的token初始化tushare，你可以在注册后通过Tushare网站获取token cac42317028fe98e755d31c9a6c4f615f995aa2a87946931c0caa90881ba2fb7
    ts.set_token('48c54212788b6a040d89de4ee5810744d936b44c2423302761f3b254')  # 请替换成你自己的token
    pro = ts.pro_api()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
    plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of minus sign '-' display as square
    # 基础数据
    start_date = '20210401'  #回测期开始日
    end_date = '20231001'   #回测期结束日
    directory ='D:\\Stock_data_fscore'  ##
    # 投资组合每日pct的dataframe
    Portfolio_Daily_Summary,one_month_sample = get_portfolio_backtest_periods_pct(start_date=start_date, end_date=end_date, directory=directory,
                                                                 rebalancing_method=get_fitst_trading_day_after_report_date,
                                                                 select_strategy=select_low_total_mv_stocks,column='total_mv',percentile_or_value=7)
    # 回测统计数据
    sys.path.append("F:\\BaiduSyncdisk\\pycharm_code\\回测框架")  # 使用目标文件夹的绝对路径替换此路径
    from backtest import run_Stock_Selection_Strategy  # 替换为你的文件名和函数名
    # 运行回测
    results =  run_Stock_Selection_Strategy(Portfolio_Daily_Summary)
    # 打印或处理结果
    print(results)


 













