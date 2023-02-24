import tushare as ts
import matplotlib.pylab as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

token = '7e0f9a0477a8b7c89481ba56e5c4b7e7667676b4ddf888143a9715d5'
pro = ts.pro_api(token)
df1 = pro.daily(ts_code='000001.SZ', start_date='20201101', end_date='20201201')

df1.head()
# df1.to_excel('股票价格.xlsx', index=False)
print(df1)

df1.set_index('trade_date', inplace=True)
df1['close'].plot(title='股票走势')

df2 = pro.daily(ts_code='000001.SZ', start_date='20091101', end_date='20201201')
from datetime import datetime

df2.to_excel('股票价格2.xlsx', index=False)
df2['trade_date'] = df2['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

plt.plot(df2['trade_date'], df2['close'])
plt.show()
