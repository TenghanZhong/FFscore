import tushare as ts
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import time
##
#pd.merge_asof() 是 Pandas 中的一个函数，用于在“最接近”的键上执行合并，而不是要求完全匹配的键。
# 这在时间序列数据中特别有用，因为您可能需要将两组数据合并在一起，这两组数据在时间点上不完全对应，但您想要找到最接近的时间点进行匹配和合并。
# 这常用于金融时间序列，其中您有不同频率的数据（例如，交易数据和季度报告）。

ts.set_token('48c54212788b6a040d89de4ee5810744d936b44c2423302761f3b254')

# 初始化Tushare API
pro = ts.pro_api()

# 定义要保存的文件路径
save_path = 'D:\\Stock_data_fscore\\'

# 如果目录不存在，创建目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 获取所有A股股票列表
all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code')

unique_stocks = all_stocks['ts_code'].unique()#获得所有股票的股票代码

def get_fitst_trading_day_after_report_date(start_date, end_date):
    '''
    获取回测期间每个报告期后第一个交易日的列表
    :param start_date: 起始日  eg: start_date = '20060401'
    :param end_date: 终止日 eg: end_date = '20160401'
    :return: 返回list格式的回测期的每个fitst_trading_day_after_report_date
    '''

    trade_cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    open_days = trade_cal[trade_cal['is_open'] == 1]
    report_dates = ['03-31', '06-30', '09-30', '12-31']
    rebalancing_days = []#存储每个调仓日期

    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for report_date in report_dates:
            date_str = pd.to_datetime(f"{year}-{report_date}")
            if date_str >= pd.to_datetime(start_date) and date_str <= pd.to_datetime(end_date):
                first_trade_date_after_report = open_days.loc[
                    pd.to_datetime(open_days['cal_date']) >= date_str, 'cal_date'].min()
                if not pd.isnull(first_trade_date_after_report):
                    formatted_date = pd.to_datetime(first_trade_date_after_report)
                    rebalancing_days.append(formatted_date)

##转换日期格式方法：formatted_date = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
##：formatted_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')

    return rebalancing_days
datetime_objects= get_fitst_trading_day_after_report_date('2005-09-01', '2023-10-30')
# 转换为日期类型
report_periods = datetime_objects


# 定义需要的字段
fina_fields = [
    'ts_code', 'end_date', 'roa','ocfps', 'ebit','grossprofit_margin','assets_turn'
]#季度或年报得到
daily_fields = [
    'ts_code', 'trade_date','total_mv','pb','close','turnover_rate','pe'
]#每日数据
balancesheet_fields=['ts_code','end_date','total_assets','total_ncl','total_cur_assets','total_cur_liab','total_hldr_eqy_exc_min_int']#季度或年报得到
cashflow_fields=['ts_code','end_date','n_cashflow_act']

code_list=[]
pb_zscore_relationships_list = []
sample_count=0

def get_data(ts_code,fina_fields,daily_fields,balancesheet_fields,cashflow_fields):
    fina_data = pro.fina_indicator(ts_code=ts_code, fields=','.join(fina_fields))
    daily_data = pro.daily_basic(ts_code=ts_code, fields=','.join(daily_fields))
    balancesheet_data = pro.balancesheet(ts_code=ts_code, fields=','.join(balancesheet_fields))
    cashflow_data = pro.cashflow(ts_code=ts_code, fields=','.join(cashflow_fields))

    # Convert date columns to datetime type
    fina_data['end_date'] = pd.to_datetime(fina_data['end_date'])
    daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])
    balancesheet_data['end_date'] = pd.to_datetime(balancesheet_data['end_date'])
    cashflow_data['end_date'] = pd.to_datetime(cashflow_data['end_date'])

    # First merge: Merge daily_data and fina_data    merge_asof一定要对已日期排序的文件使用
    merged_1 = pd.merge_asof(daily_data.sort_values('trade_date',ascending=True), fina_data.sort_values('end_date'),
                             by='ts_code', left_on='trade_date', right_on='end_date')

    # Second merge: Merge the result with balancesheet_data
    merged_2 = pd.merge_asof(merged_1.sort_values('trade_date'), balancesheet_data.sort_values('end_date'),
                             by='ts_code', left_on='trade_date', right_on='end_date')

    # Third merge: Merge the result with income_data
    merged_final = pd.merge_asof(merged_2.sort_values('trade_date'), cashflow_data.sort_values('end_date'),
                                 by='ts_code', left_on='trade_date', right_on='end_date')
    #对于 merged_2 中的每个 trade_date，它会寻找 cashflow_data 中最近的但不晚于该 trade_date 的 end_date。
    return merged_final

def select_period(data,start_date,end_date):
    '''

    :param data: dataframe
    :param start_date: '2005-12-30'
    :param end_date: '2016-04-30'
    :return:
    '''
    # 创建一个布尔掩码来过滤出特定日期范围内的数据
    mask_period = (data['trade_date'] >= start_date) & (data['trade_date'] <= end_date)
    # 应用掩码以获取所需的数据
    filtered_data = merged_final[mask_period]
    filtered_data = filtered_data.dropna()

    return filtered_data

def count_fscore(filtered_data,first_trade_date_after_report):
    mask = (filtered_data['trade_date'] == first_trade_date_after_report)
    filtered_data.loc[mask, 'cfo'] = filtered_data['n_cashflow_act'] / filtered_data['total_assets']
    filtered_data.loc[mask, 'roa_change'] = filtered_data['roa'] - filtered_data['roa'].shift(1)
    filtered_data.loc[mask, 'accural'] = filtered_data['cfo'] - filtered_data['roa']
    filtered_data.loc[mask, 'lever_change'] = (filtered_data['total_ncl'] / (
            filtered_data['total_assets'] - filtered_data['total_cur_assets'])) - \
                                              (filtered_data['total_ncl'] / (
                                                      filtered_data['total_assets'] - filtered_data[
                                                  'total_cur_assets'])).shift(1)
    filtered_data.loc[mask, 'liquid_change'] = (filtered_data['total_cur_assets'] / filtered_data[
        'total_cur_liab']) - \
                                               (filtered_data['total_cur_assets'] / filtered_data[
                                                   'total_cur_liab']).shift(1)
    filtered_data.loc[mask, 'equity_change'] = filtered_data['total_hldr_eqy_exc_min_int'] - filtered_data[
        'total_hldr_eqy_exc_min_int'].shift(1)
    filtered_data.loc[mask, 'margin_change'] = filtered_data['grossprofit_margin'] - filtered_data[
        'grossprofit_margin'].shift(1)
    filtered_data.loc[mask, 'assets_turn_change'] = filtered_data['assets_turn'] - filtered_data[
        'assets_turn'].shift(1)

    # 在报告期内计算F分数 计算每个条件的得分
    scores = (np.where(filtered_data['roa'] > 0, 1, 0) +
              np.where(filtered_data['cfo'] > 0, 1, 0) +
              np.where(filtered_data['roa_change'] > 0, 1, 0) +
              np.where(filtered_data['accural'] > 0, 1, 0) +
              np.where(filtered_data['lever_change'] < 0, 1, 0) +
              np.where(filtered_data['liquid_change'] > 0, 1, 0) +
              np.where(filtered_data['equity_change'] > 0, 1, 0) +
              np.where(filtered_data['margin_change'] > 0, 1, 0) +
              np.where(filtered_data['assets_turn_change'] > 0, 1, 0))
    # 仅将得分分配给标记的行
    filtered_data.loc[mask, 'f_score'] = scores[mask]  # 索引，score返回的series类型
    return filtered_data


# 对于每一只股票
for ts_code in tqdm(unique_stocks, desc='Processing stocks', unit='stock'):
    try:
        merged_final=get_data(ts_code,fina_fields,daily_fields,balancesheet_fields,cashflow_fields)
        filtered_data=select_period(merged_final,'2005-09-01','2023-10-30')

        if filtered_data.shape[0]==0:#如果没有数据，0行
            print("{} is Empty.".format(ts_code))
            continue
        '''
        #cfo计算 经营现金流率  为正，分数+1
        filtered_data['cfo']=filtered_data['n_cashflow_act']/filtered_data['total_assets']

        #roa同比变化率 变化为正,分数加1
        filtered_data['roa_change']=filtered_data['roa']-filtered_data['roa'].shift(1)

        #accrual 公司自然增长获利 =cfo-roa  变化为正,分数加1
        filtered_data['accural']=filtered_data['cfo']-filtered_data['roa']

        #杠杆变化 通过所有非流动负债合计除以非流动性资产计算公司财务杠杆， 变化为负数时△LEVER=1
        filtered_data['lever_change'] = (filtered_data['total_ncl']/ (filtered_data['total_assets']-filtered_data[
            'total_cur_assets']))-(filtered_data['total_ncl']/(filtered_data['total_assets']-filtered_data[
            'total_cur_assets'])).shift(1)

        #流动性变化（△LIQUID）：流动性通过流动比率，即流动资产除以流动负债计算 变化为正数时+1
        filtered_data['liquid_change']=(filtered_data['total_cur_assets']/filtered_data['total_cur_liab']
                                        )-(filtered_data['total_cur_assets']/filtered_data['total_cur_liab']).shift(1)

        #是否发行普通股权  变化为正数时，EQ_OFFER=1.否则为0。
        filtered_data['equity_change']=filtered_data['total_hldr_eqy_exc_min_int']-filtered_data['total_hldr_eqy_exc_min_int'].shift(1)

        #毛利率变化  变化为正数时△MARGIN =1，否则为0。
        filtered_data['margin_change']=filtered_data['grossprofit_margin']-filtered_data['grossprofit_margin'].shift(1)
        #资产周转率变化 资产周转率变化为正数时△TURN =1，否则为0。
        filtered_data['assets_turn_change']=filtered_data['assets_turn']-filtered_data['assets_turn'].shift(1)


        #计算fscore
        ###PS:total_mv以（万元）为单位
        filtered_data['f_score']= pd.DataFrame(np.where(filtered_data['roa']>0,1,0)+
                                               np.where(filtered_data['cfo']>0,1,0)+np.where(filtered_data['roa_change']>0,1,0)+
                                               np.where(filtered_data['accural']>0,1,0)+np.where(filtered_data['lever_change']<0,1,0)+
                                               np.where(filtered_data['liquid_change']>0,1,0)+np.where(filtered_data['equity_change']>0,1,0)+
                                               np.where(filtered_data['margin_change']>0,1,0)+np.where(filtered_data['assets_turn_change']>0,1,0))
        '''

        filtered_data['pct']=filtered_data['close']/filtered_data['close'].shift(1)-1
        filtered_data=filtered_data.dropna()
        #初始化列
        columns_to_calculate = ['cfo', 'roa_change', 'accural', 'lever_change', 'liquid_change',
                                'equity_change', 'margin_change', 'assets_turn_change', 'f_score']
        for column in columns_to_calculate:
            filtered_data[column] = np.nan  #np.zeros不行，是因为后面要利用fillna(method='ffill')来进行填充

        # 只在报告期内计算指标
        # 对于每个报告期，找到第一个交易日，并仅在那一天进行计算
        for report_date in report_periods:
            # 查找报告日期之后的第一个交易日 data.loc[布尔类型索引,值].min()
            first_trade_date_after_report = filtered_data.loc[
                filtered_data['trade_date'] >= report_date, 'trade_date'].min()
            # 如果存在这样的交易日，就在那一天进行计算 if not pd.isnull(day)
            if not pd.isnull(first_trade_date_after_report):
                #掩码 data.loc[data['score']==1]=data[data['score']==1]
                filtered_data=count_fscore(filtered_data,first_trade_date_after_report)


        # 前向填充指标到下一个报告期  #.fillna(method='ffill')向下填充
        filtered_data[columns_to_calculate] = filtered_data[columns_to_calculate].fillna(method='ffill')
        filtered_data=filtered_data.dropna()

        if filtered_data.shape[0]==0:#如果没有数据，0行
            print("{} is Empty.".format(ts_code))
            continue

        code_list.append(ts_code)
        print('已完成：{}'.format(ts_code))

        file_path = os.path.join(save_path, f"{ts_code}.csv.gz") #os.path.join()
        filtered_data.to_csv(file_path, index=False, compression='gzip')

    except Exception as e:
        print(f"Error processing {ts_code}: {str(e)}")


code_list=pd.DataFrame(code_list,columns=['Ashare_codelist'])
file_path2 = os.path.join(save_path, f"code_list.csv")
code_list.to_csv(file_path2, index=False)

print("Data retrieval and merging complete.")
