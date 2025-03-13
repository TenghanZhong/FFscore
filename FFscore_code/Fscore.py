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
save_path = 'D:\\Stock_data_ashare\\'

# 如果目录不存在，创建目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 获取所有A股股票列表
all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code')

unique_stocks = all_stocks['ts_code'].unique()
# 定义需要的字段
fina_fields = [
    'ts_code', 'end_date', 'roa', 'ocfps', 'basic_eps_yoy', 'longdeb_to_debt',
    'current_ratio', 'grossprofit_margin',
    'assets_turn', 'ebit_ps', 'netdebt'
]

daily_fields = [
    'ts_code', 'total_share','trade_date', 'pe', 'total_mv','close','pb'
]

code_list=[]
# 对于每一只股票
for ts_code in tqdm(unique_stocks, desc='Processing stocks', unit='stock'):
    code_list.append(ts_code)
    try:
        fina_data = pro.fina_indicator(ts_code=ts_code, fields=','.join(fina_fields))
        daily_data = pro.daily_basic(ts_code=ts_code, fields=','.join(daily_fields))

        # Convert date columns to datetime type
        fina_data['end_date'] = pd.to_datetime(fina_data['end_date'])
        daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])

        # Merge the data using merge_asof to align the dates
        merged_data = pd.merge_asof(daily_data.sort_values('trade_date'), fina_data.sort_values('end_date'),
                                    by='ts_code', left_on='trade_date', right_on='end_date')
        merged_data['netdebt_total_mv'] = merged_data['netdebt'] + merged_data['total_mv']

        file_path = os.path.join(save_path, f"{ts_code}.csv.gz")
        merged_data.to_csv(file_path, index=False, compression='gzip')
        print(f"Data for {ts_code} saved to {file_path}")
    except Exception as e:
        print(f"Error processing {ts_code}: {str(e)}")


code_list=pd.DataFrame(code_list,columns=['Ashare_codelist'])
file_path = os.path.join(save_path, "{ts_code}.csv")
code_list.to_csv(file_path, index=False)

print("Data retrieval and merging complete.")
print(code_list)