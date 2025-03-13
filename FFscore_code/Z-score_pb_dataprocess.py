import tushare as ts
import pandas as pd
import os
from tqdm import tqdm
import time
##
#pd.merge_asof() 是 Pandas 中的一个函数，用于在“最接近”的键上执行合并，而不是要求完全匹配的键。
# 这在时间序列数据中特别有用，因为您可能需要将两组数据合并在一起，这两组数据在时间点上不完全对应，但您想要找到最接近的时间点进行匹配和合并。
# 这常用于金融时间序列，其中您有不同频率的数据（例如，交易数据和季度报告）。

ts.set_token('48c54212788b6a040d89de4ee5810744d936b44c2423302761f3b254')

# 初始化Tushare API
pro = ts.pro_api()

# 定义要保存的文件路径
save_path = 'D:\\Stock_data_zscore\\'

# 如果目录不存在，创建目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 获取所有A股股票列表
all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code')

unique_stocks = all_stocks['ts_code'].unique()
# 定义需要的字段
fina_fields = [
    'ts_code', 'end_date', 'retained_earnings','working_capital', 'ebit',
]#季度或年报得到
daily_fields = [
    'ts_code', 'trade_date','total_mv','pb','close','turnover_rate','pe'
]#每日数据
balancesheet_fields=['ts_code','end_date','total_assets','total_liab']#季度或年报得到
income_fields=['ts_code','end_date','revenue']#季度或年报得到

code_list=[]
pb_zscore_relationships_list = []
sample_count=0
# 对于每一只股票
for ts_code in tqdm(unique_stocks, desc='Processing stocks', unit='stock'):
    try:
        fina_data = pro.fina_indicator(ts_code=ts_code, fields=','.join(fina_fields))
        daily_data = pro.daily_basic(ts_code=ts_code, fields=','.join(daily_fields))
        balancesheet_data=pro.balancesheet(ts_code=ts_code, fields=','.join(balancesheet_fields))
        income_data=pro.income(ts_code=ts_code, fields=','.join(income_fields))

        # Convert date columns to datetime type
        fina_data['end_date'] = pd.to_datetime(fina_data['end_date'])
        daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])
        balancesheet_data['end_date'] = pd.to_datetime(balancesheet_data['end_date'])
        income_data['end_date'] = pd.to_datetime(income_data['end_date'])

        # First merge: Merge daily_data and fina_data    merge_asof一定要对已日期排序的文件使用
        merged_1 = pd.merge_asof(daily_data.sort_values('trade_date'), fina_data.sort_values('end_date'),
                                 by='ts_code', left_on='trade_date', right_on='end_date')

        # Second merge: Merge the result with balancesheet_data
        merged_2 = pd.merge_asof(merged_1.sort_values('trade_date'), balancesheet_data.sort_values('end_date'),
                                 by='ts_code', left_on='trade_date', right_on='end_date')

        # Third merge: Merge the result with income_data
        merged_final = pd.merge_asof(merged_2.sort_values('trade_date'), income_data.sort_values('end_date'),
                                     by='ts_code', left_on='trade_date', right_on='end_date')

        # 创建一个布尔掩码来过滤出特定日期范围内的数据
        mask_period = (merged_final['trade_date'] >= '2006-04-01') & (merged_final['trade_date'] <= '2016-04-01')
        # 应用掩码以获取所需的数据
        filtered_data = merged_final[mask_period]

        # 使用drop_duplicates方法保留每个end_date的第一个条目,即报告期的数据 drop_duplicates(subset='end_date')
        filtered_data=filtered_data.drop_duplicates(subset='end_date') #drop_duplocates
        filtered_data.dropna(inplace=True)#在报告期若有空值，删除当期样本---研报原文



        if filtered_data.shape[0]==0:#如果没有数据，0行
            print("{} is Empty.".format(ts_code))
            continue

        #计算Zscore
        #当 Z 值大于 2.675 时,表示企业的财务状况良好；当 Z 值小于 1.81 时,则表明企业陷入财务困境且
        #有较高的破产可能性；当 Z 值介于 1.81 和 2.675 之间时被称之为灰色区域,表示企业的财
        #务状况极为不稳定。
        ###PS:total_mv以（万元）为单位
        filtered_data['z_score']=1.2*(filtered_data['working_capital']/filtered_data['total_assets'])+1.4*(
                filtered_data['retained_earnings']/filtered_data['total_assets'])+3.3*(
                filtered_data['ebit']/filtered_data['total_assets'])+0.6*(
                (filtered_data['total_mv']*10000)/filtered_data['total_liab'])+0.999*(
                filtered_data['revenue']/filtered_data['total_assets'])

        # 创建一个布尔掩码来过滤zscore>300或zscore<-100的异常值并创建一个布尔掩码来过滤pb>20或pb<0的异常值,
        # 也可以用query索引条件（更简洁）
        filtered_data = filtered_data.query('0 <= pb <= 20 and -100 <= z_score <= 300')

        pb_zscore_relationships_list.append(filtered_data[['ts_code','end_date','close','pb', 'z_score']])
        code_list.append(ts_code)
        sample_count+=filtered_data.shape[0]
        print('样本数额为：{}'.format(sample_count))

        file_path = os.path.join(save_path, f"{ts_code}.csv.gz") #os.path.join()
        filtered_data.to_csv(file_path, index=False, compression='gzip')

    except Exception as e:
        print(f"Error processing {ts_code}: {str(e)}")



pb_zscore_relationships = pd.concat(pb_zscore_relationships_list, ignore_index=True, axis=0)

code_list=pd.DataFrame(code_list,columns=['Ashare_codelist'])
file_path1 = os.path.join(save_path, f"pb_zscore_relationships.csv")
file_path2 = os.path.join(save_path, f"code_list.csv")
pb_zscore_relationships.to_csv(file_path1, index=False)#转为csv文件
code_list.to_csv(file_path2, index=False)

print("Data retrieval and merging complete.")
