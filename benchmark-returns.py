#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Author : mda@pyrosome.com
    Date: 2017-12-12
    Plotting stocks benchmark returns
'''


import matplotlib.pyplot as plt
import pandas as pd
import quandl
from datetime import date
import config

quandl.ApiConfig.api_key = config.API_CONFIG_KEY

end_date = today = date.today()
start_date = date(today.year - 1, today.month, today.day).isoformat()


def benchmark(stock, stocklist, start_date, end_date):
    '''
    Plots the returns of a stock against a list of stocks
    stock1 : main stock
    stocklist: the stock list of at least 1 element
    start_date/end_date: int 2014-11-01
    '''

    stockdf = quandl.get(stock, start_date=start_date,
                         end_date=end_date)

    benchstock = pd.DataFrame({stock: stockdf['Close']})

    try:
        for stock in stocklist:
            stockR = quandl.get(stock,
                                start_date=start_date,
                                end_date=end_date)
            benchstock = benchstock.append(
                {stock: stockR['Close']}, ignore_index=True)

    except Exception as e:
        print(e)

        '''
        Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend',
        'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low',
        'Adj. Close', 'Adj. Volume'], dtype='object')
        '''
        # stockdf.plot(secondary_y=[benchstock], grid=True)
        # shift moves dates back by 1.
        # stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    print(benchstock)
    try:
        stock_return = benchstock.apply(lambda x: x / x[0])
        stock_return.plot(grid=True).axhline(y=1, color="black", lw=2)

        plt.show()

    except Exception as e:
        print(e)


benchmark('WIKI/INTC', ('WIKI/NVDA'), start_date, end_date)
